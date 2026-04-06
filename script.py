#!/usr/bin/env python3
"""
github_scorer.py
Pure stats-based GitHub profile scorer.
Measures demonstrable output signals — not idea quality.

Usage:
    python github_scorer.py <username> [--token TOKEN] [--include-forks] [--output both|json|terminal]
"""

import argparse
import json
import sys
import re
from datetime import datetime, timezone
from collections import defaultdict

try:
    from github import Github, GithubException
except ImportError:
    print("Missing dependency: pip install PyGitHub rich")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

NOISE_PATTERNS = re.compile(
    r"^(fix typo|typo|wip|update readme|minor|cleanup|fmt|format|lint|merge|"
    r"initial commit|init|add files|first commit|test|bump version|bump|"
    r"update|updates|misc|refactor formatting|whitespace)",
    re.IGNORECASE
)

TEST_PATTERNS = ["test_", "_test.", ".test.", ".spec.", "__tests__", "/tests/", "/test/", "/spec/"]
DOCS_PATTERNS = ["README", "docs/", "CHANGELOG", "CONTRIBUTING", "wiki"]

WEIGHTS = {
    "velocity_score":      0.20,
    "trajectory_score":    0.20,
    "shipping_score":      0.20,
    "quality_score":       0.20,
    "consistency_score":   0.10,
    "originality_score":   0.10,
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc)

def months_ago(dt, n):
    """Return commits from dt within last n months."""
    cutoff = now().replace(month=((now().month - n - 1) % 12) + 1)
    return dt >= cutoff

def is_noise(msg: str) -> bool:
    return bool(NOISE_PATTERNS.match(msg.strip()))

def has_tests(repo) -> bool:
    try:
        contents = repo.get_git_tree("HEAD", recursive=True)
        for f in contents.tree:
            if any(p in f.path for p in TEST_PATTERNS):
                return True
    except Exception:
        pass
    return False

def has_docs(repo) -> bool:
    try:
        contents = repo.get_git_tree("HEAD", recursive=True)
        for f in contents.tree:
            if any(p in f.path for p in DOCS_PATTERNS):
                return True
    except Exception:
        pass
    return False

def get_commit_months(repo, max_commits=300):
    """Return list of (month_key, is_substantive) tuples."""
    results = []
    try:
        commits = repo.get_commits()
        for i, c in enumerate(commits):
            if i >= max_commits:
                break
            msg = c.commit.message.split("\n")[0]
            date = c.commit.author.date.replace(tzinfo=timezone.utc)
            month_key = date.strftime("%Y-%m")
            results.append((month_key, not is_noise(msg), date))
    except Exception:
        pass
    return results

def is_published(repo):
    """Check if repo has releases or tags."""
    try:
        releases = list(repo.get_releases())
        if releases:
            return True
        tags = list(repo.get_tags())
        return len(tags) > 0
    except Exception:
        return False

# ---------------------------------------------------------------------------
# SCORING FUNCTIONS  (each returns 0.0 – 100.0)
# ---------------------------------------------------------------------------

def score_velocity(all_commits_by_month) -> tuple[float, dict]:
    """Average substantive commits per active month."""
    if not all_commits_by_month:
        return 0.0, {}
    
    total_substantive = sum(v["substantive"] for v in all_commits_by_month.values())
    active_months = sum(1 for v in all_commits_by_month.values() if v["total"] > 0)
    
    if active_months == 0:
        return 0.0, {}
    
    avg = total_substantive / active_months
    # ~5 substantive commits/month = 60, ~15 = 100
    score = min(100, (avg / 15) * 100)
    return round(score, 1), {"avg_substantive_per_month": round(avg, 1), "active_months": active_months}

def score_trajectory(all_commits_by_month) -> tuple[float, dict]:
    """Is activity increasing over time? Compare recent 3mo vs prior 3mo."""
    if not all_commits_by_month:
        return 50.0, {}
    
    sorted_months = sorted(all_commits_by_month.keys())
    if len(sorted_months) < 2:
        return 50.0, {}
    
    recent = sorted_months[-3:]
    prior = sorted_months[-6:-3]
    
    recent_sub = sum(all_commits_by_month[m]["substantive"] for m in recent)
    prior_sub = sum(all_commits_by_month[m]["substantive"] for m in prior) if prior else 0
    
    if prior_sub == 0:
        trajectory = 1.0 if recent_sub > 0 else 0.0
    else:
        trajectory = recent_sub / prior_sub
    
    # 1.0 = flat = 50, 2.0+ = growing = 100, 0 = dying = 0
    score = min(100, trajectory * 50)
    return round(score, 1), {"recent_3mo": recent_sub, "prior_3mo": prior_sub, "ratio": round(trajectory, 2)}

def score_shipping(repos) -> tuple[float, dict]:
    """Ratio of repos that actually shipped (release/tag/published package)."""
    if not repos:
        return 0.0, {}
    
    shipped = sum(1 for r in repos if r["published"])
    ratio = shipped / len(repos)
    score = min(100, ratio * 100 * 1.5)  # 67% shipped = 100
    return round(score, 1), {"shipped": shipped, "total": len(repos), "ratio": round(ratio, 2)}

def score_quality(repos) -> tuple[float, dict]:
    """Proxy quality: tests, docs, description, non-trivial size."""
    if not repos:
        return 0.0, {}
    
    scores = []
    for r in repos:
        s = 0
        if r["has_tests"]: s += 35
        if r["has_docs"]: s += 25
        if r["description"]: s += 15
        if r["stars"] >= 1: s += 10
        if r["size_kb"] > 50: s += 15
        scores.append(s)
    
    avg = sum(scores) / len(scores)
    return round(avg, 1), {"repos_with_tests": sum(1 for r in repos if r["has_tests"]),
                           "repos_with_docs": sum(1 for r in repos if r["has_docs"])}

def score_consistency(all_commits_by_month) -> tuple[float, dict]:
    """How consistent is activity — penalize long gaps."""
    if not all_commits_by_month:
        return 0.0, {}
    
    sorted_months = sorted(all_commits_by_month.keys())
    if len(sorted_months) < 2:
        return 50.0, {}
    
    # Find gaps: months between first and last with zero activity
    from datetime import date
    first = datetime.strptime(sorted_months[0], "%Y-%m")
    last = datetime.strptime(sorted_months[-1], "%Y-%m")
    
    total_months = (last.year - first.year) * 12 + (last.month - first.month) + 1
    active = len([m for m in all_commits_by_month.values() if m["total"] > 0])
    
    ratio = active / total_months if total_months > 0 else 0
    score = min(100, ratio * 100)
    return round(score, 1), {"active_months": active, "total_span_months": total_months, "fill_rate": round(ratio, 2)}

def score_originality(repos, include_forks) -> tuple[float, dict]:
    """Ratio of original repos vs forks. If forks excluded, reward language diversity."""
    if not repos:
        return 50.0, {}
    
    original = sum(1 for r in repos if not r["fork"])
    ratio = original / len(repos)
    
    langs = set(r["language"] for r in repos if r["language"])
    lang_diversity = min(1.0, len(langs) / 5)  # 5 languages = max diversity credit
    
    score = (ratio * 70) + (lang_diversity * 30)
    return round(min(100, score), 1), {"original_repos": original, "languages": list(langs)}

# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyze(username: str, token: str = None, include_forks: bool = False) -> dict:
    g = Github(token) if token else Github()
    
    try:
        user = g.get_user(username)
    except GithubException as e:
        print(f"Error fetching user '{username}': {e}")
        sys.exit(1)
    
    if RICH:
        console.print(f"\n[bold cyan]Fetching repos for [white]{username}[/white]...[/bold cyan]")
    else:
        print(f"Fetching repos for {username}...")
    
    all_repos_raw = list(user.get_repos())
    repos_raw = [r for r in all_repos_raw if include_forks or not r.fork]
    
    if not repos_raw:
        print("No repos found with current settings.")
        sys.exit(0)
    
    repos = []
    all_commits_by_month = defaultdict(lambda: {"total": 0, "substantive": 0})
    
    for i, repo in enumerate(repos_raw):
        if RICH:
            console.print(f"  [{i+1}/{len(repos_raw)}] [dim]{repo.name}[/dim]")
        else:
            print(f"  [{i+1}/{len(repos_raw)}] {repo.name}")
        
        commit_data = get_commit_months(repo)
        
        for month_key, is_sub, _ in commit_data:
            all_commits_by_month[month_key]["total"] += 1
            if is_sub:
                all_commits_by_month[month_key]["substantive"] += 1
        
        repos.append({
            "name": repo.name,
            "fork": repo.fork,
            "description": bool(repo.description),
            "stars": repo.stargazers_count,
            "language": repo.language,
            "size_kb": repo.size,
            "has_tests": has_tests(repo),
            "has_docs": has_docs(repo),
            "published": is_published(repo),
            "created_at": repo.created_at.isoformat(),
        })
    
    # Score each dimension
    vel_score, vel_detail = score_velocity(all_commits_by_month)
    traj_score, traj_detail = score_trajectory(all_commits_by_month)
    ship_score, ship_detail = score_shipping(repos)
    qual_score, qual_detail = score_quality(repos)
    cons_score, cons_detail = score_consistency(all_commits_by_month)
    orig_score, orig_detail = score_originality(repos, include_forks)
    
    dimension_scores = {
        "velocity_score": vel_score,
        "trajectory_score": traj_score,
        "shipping_score": ship_score,
        "quality_score": qual_score,
        "consistency_score": cons_score,
        "originality_score": orig_score,
    }
    
    overall = sum(dimension_scores[k] * WEIGHTS[k] for k in WEIGHTS)
    
    return {
        "username": username,
        "analyzed_at": now().isoformat(),
        "config": {"include_forks": include_forks, "repos_analyzed": len(repos)},
        "overall_score": round(overall, 1),
        "dimensions": dimension_scores,
        "details": {
            "velocity": vel_detail,
            "trajectory": traj_detail,
            "shipping": ship_detail,
            "quality": qual_detail,
            "consistency": cons_detail,
            "originality": orig_detail,
        },
        "repos": repos,
    }

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def bar(score: float, width: int = 20) -> str:
    filled = int((score / 100) * width)
    return "█" * filled + "░" * (width - filled)

def color(score: float) -> str:
    if score >= 75: return "green"
    if score >= 50: return "yellow"
    return "red"

def print_terminal(result: dict):
    if not RICH:
        # Plain fallback
        print(f"\n=== GitHub Score: {result['username']} ===")
        print(f"Overall: {result['overall_score']}/100")
        for k, v in result["dimensions"].items():
            print(f"  {k}: {v}")
        return
    
    console.print()
    console.rule(f"[bold]GitHub Scorer — [cyan]{result['username']}[/cyan]")
    console.print()
    
    # Overall
    overall = result["overall_score"]
    col = color(overall)
    console.print(Panel(
        f"[bold {col}]{overall} / 100[/bold {col}]\n[dim]{bar(overall, 30)}[/dim]",
        title="[bold]Overall Score[/bold]",
        expand=False
    ))
    console.print()
    
    # Dimensions table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold dim")
    table.add_column("Dimension", style="bold", min_width=20)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("Bar", min_width=22)
    table.add_column("Weight", justify="right", min_width=6)
    
    labels = {
        "velocity_score":    "Commit Velocity",
        "trajectory_score":  "Trajectory (Growth)",
        "shipping_score":    "Shipping Rate",
        "quality_score":     "Quality Proxies",
        "consistency_score": "Consistency",
        "originality_score": "Originality",
    }
    
    for key, label in labels.items():
        score = result["dimensions"][key]
        col = color(score)
        weight_pct = f"{int(WEIGHTS[key]*100)}%"
        table.add_row(
            label,
            f"[{col}]{score}[/{col}]",
            f"[{col}]{bar(score)}[/{col}]",
            f"[dim]{weight_pct}[/dim]"
        )
    
    console.print(table)
    console.print()
    
    # Details
    details = result["details"]
    console.print("[bold dim]Details[/bold dim]")
    
    vel = details["velocity"]
    if vel:
        console.print(f"  Velocity   → {vel.get('avg_substantive_per_month', '?')} substantive commits/month "
                      f"over {vel.get('active_months', '?')} active months")
    
    traj = details["trajectory"]
    if traj:
        direction = "↑ growing" if traj.get("ratio", 1) > 1 else ("↓ declining" if traj.get("ratio", 1) < 1 else "→ flat")
        console.print(f"  Trajectory → recent 3mo: {traj.get('recent_3mo')} vs prior 3mo: {traj.get('prior_3mo')} — [bold]{direction}[/bold]")
    
    ship = details["shipping"]
    if ship:
        console.print(f"  Shipping   → {ship.get('shipped')}/{ship.get('total')} repos have releases or tags")
    
    qual = details["quality"]
    if qual:
        console.print(f"  Quality    → {qual.get('repos_with_tests')} repos with tests, {qual.get('repos_with_docs')} with docs")
    
    cons = details["consistency"]
    if cons:
        console.print(f"  Consistency→ active {cons.get('active_months')}/{cons.get('total_span_months')} months "
                      f"({int(cons.get('fill_rate',0)*100)}% fill rate)")
    
    orig = details["originality"]
    if orig:
        langs = ", ".join(orig.get("languages", [])) or "none detected"
        console.print(f"  Originality→ {orig.get('original_repos')} original repos | languages: {langs}")
    
    console.print()
    console.print(f"[dim]Repos analyzed: {result['config']['repos_analyzed']} "
                  f"({'including' if result['config']['include_forks'] else 'excluding'} forks) "
                  f"| {result['analyzed_at'][:10]}[/dim]")
    console.print()

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score a GitHub profile on pure output signals.")
    parser.add_argument("username", help="GitHub username to analyze")
    parser.add_argument("--token", default=None, help="GitHub personal access token (avoids rate limits)")
    parser.add_argument("--include-forks", action="store_true", help="Include forked repos in analysis")
    parser.add_argument("--output", choices=["terminal", "json", "both"], default="both",
                        help="Output format (default: both)")
    args = parser.parse_args()
    
    result = analyze(args.username, token=args.token, include_forks=args.include_forks)
    
    if args.output in ("terminal", "both"):
        print_terminal(result)
    
    if args.output in ("json", "both"):
        fname = f"{args.username}_github_score.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        if RICH:
            console.print(f"[dim]JSON saved → [bold]{fname}[/bold][/dim]\n")
        else:
            print(f"JSON saved → {fname}")

if __name__ == "__main__":
    main()
