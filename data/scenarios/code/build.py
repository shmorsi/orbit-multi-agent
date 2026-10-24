"""Write the frozen Orbit Code mini repos.

Running this again rewrites the same files. The scorer trusts the manifest hash,
not this script. Do not change a repo, a hidden test, or the manifest by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

CASES = [
    {
        "id": "sum-inclusive",
        "repo": "sum_inclusive",
        "entry": "sum_inclusive.py",
        "issue": "sum_to(n) must add every integer from 1 through n, including n.",
        "buggy": "def sum_to(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total\n",
        "oracle": "def sum_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total\n",
        "wrong": "def sum_to(n):\n    total = 0\n    for i in range(0, n):\n        total += i\n    return total\n",
        "visible": "from sum_inclusive import sum_to\n\n\ndef test_visible():\n    assert sum_to(5) == 15\n",
        "hidden": "from sum_inclusive import sum_to\n\n\n# orbit-hidden\ndef test_hidden():\n    assert sum_to(1) == 1\n    assert sum_to(10) == 55\n",
    },
    {
        "id": "palindrome-fold",
        "repo": "palindrome_fold",
        "entry": "palindrome_fold.py",
        "issue": "is_palindrome must ignore letter case and spaces.",
        "buggy": "def is_palindrome(text):\n    cleaned = text.replace(' ', '')\n    return cleaned == cleaned[::-1]\n",
        "oracle": (
            "def is_palindrome(text):\n"
            "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        "visible": "from palindrome_fold import is_palindrome\n\n\ndef test_visible():\n    assert is_palindrome('Racecar') is True\n",
        "hidden": (
            "from palindrome_fold import is_palindrome\n\n\n"
            "# orbit-hidden\n"
            "def test_hidden():\n"
            "    assert is_palindrome('A man a plan a canal Panama') is True\n"
            "    assert is_palindrome('orbit') is False\n"
        ),
    },
    {
        "id": "mean-empty",
        "repo": "mean_empty",
        "entry": "mean_empty.py",
        "issue": "mean of an empty list must be None. mean of numbers must be their average.",
        "buggy": "def mean(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n",
        "oracle": "def mean(values):\n    if not values:\n        return None\n    return sum(values) / len(values)\n",
        "visible": "from mean_empty import mean\n\n\ndef test_visible():\n    assert mean([]) is None\n",
        "hidden": (
            "from mean_empty import mean\n\n\n"
            "# orbit-hidden\n"
            "def test_hidden():\n"
            "    assert mean([]) is None\n"
            "    assert mean([2, 4]) == 3\n"
        ),
    },
    {
        "id": "tax-half-up",
        "repo": "tax_half_up",
        "entry": "tax_half_up.py",
        "issue": "tax_cents(price_cents, rate_bps) must round half up to the nearest cent. rate_bps is basis points.",
        "buggy": "def tax_cents(price_cents, rate_bps):\n    return price_cents * rate_bps // 10000\n",
        "oracle": "def tax_cents(price_cents, rate_bps):\n    return (price_cents * rate_bps + 5000) // 10000\n",
        "visible": "from tax_half_up import tax_cents\n\n\ndef test_visible():\n    assert tax_cents(100, 150) == 2\n",
        "hidden": "from tax_half_up import tax_cents\n\n\n# orbit-hidden\ndef test_hidden():\n    assert tax_cents(1, 5000) == 1\n    assert tax_cents(200, 250) == 5\n",
    },
    {
        "id": "csv-keep-tail",
        "repo": "csv_keep_tail",
        "entry": "csv_keep_tail.py",
        "issue": "parse_row must keep every comma-separated field, including the last one.",
        "buggy": "def parse_row(line):\n    return line.split(',')[:-1]\n",
        "oracle": "def parse_row(line):\n    return line.split(',')\n",
        "visible": "from csv_keep_tail import parse_row\n\n\ndef test_visible():\n    assert parse_row('a,b,c') == ['a', 'b', 'c']\n",
        "hidden": "from csv_keep_tail import parse_row\n\n\n# orbit-hidden\ndef test_hidden():\n    assert parse_row('only') == ['only']\n    assert parse_row('x,y') == ['x', 'y']\n",
    },
    {
        "id": "stack-pop",
        "repo": "stack_pop",
        "entry": "stack_pop.py",
        "issue": "pop must remove and return the last item, not the first.",
        "buggy": "def pop(items):\n    return items.pop(0)\n",
        "oracle": "def pop(items):\n    return items.pop()\n",
        "visible": "from stack_pop import pop\n\n\ndef test_visible():\n    items = [1, 2, 3]\n    assert pop(items) == 3\n    assert items == [1, 2]\n",
        "hidden": "from stack_pop import pop\n\n\n# orbit-hidden\ndef test_hidden():\n    items = [4, 9]\n    assert pop(items) == 9\n    assert items == [4]\n",
    },
    {
        "id": "parse-mdy",
        "repo": "parse_mdy",
        "entry": "parse_mdy.py",
        "issue": "parse_mdy reads month/day/year and returns a tuple (year, month, day).",
        "buggy": "def parse_mdy(text):\n    month, day, year = text.split('/')\n    return (int(year), int(day), int(month))\n",
        "oracle": "def parse_mdy(text):\n    month, day, year = text.split('/')\n    return (int(year), int(month), int(day))\n",
        "visible": "from parse_mdy import parse_mdy\n\n\ndef test_visible():\n    assert parse_mdy('03/15/2024') == (2024, 3, 15)\n",
        "hidden": "from parse_mdy import parse_mdy\n\n\n# orbit-hidden\ndef test_hidden():\n    assert parse_mdy('12/01/1999') == (1999, 12, 1)\n",
    },
    {
        "id": "binary-index",
        "repo": "binary_index",
        "entry": "binary_index.py",
        "issue": "find must return the index of target in a sorted list, or -1 when it is absent.",
        "buggy": (
            "def find(items, target):\n"
            "    for index, item in enumerate(items):\n"
            "        if item == target:\n"
            "            return index + 1\n"
            "    return -1\n"
        ),
        "oracle": (
            "def find(items, target):\n"
            "    for index, item in enumerate(items):\n"
            "        if item == target:\n"
            "            return index\n"
            "    return -1\n"
        ),
        "visible": "from binary_index import find\n\n\ndef test_visible():\n    assert find([2, 4, 6], 4) == 1\n",
        "hidden": "from binary_index import find\n\n\n# orbit-hidden\ndef test_hidden():\n    assert find([2, 4, 6], 2) == 0\n    assert find([2, 4, 6], 9) == -1\n",
    },
    {
        "id": "discount-once",
        "repo": "discount_once",
        "entry": "discount_once.py",
        "issue": "apply(price, percent) must take the percent off once.",
        "buggy": (
            "def apply(price, percent):\n"
            "    once = price * (100 - percent) / 100\n"
            "    return once * (100 - percent) / 100\n"
        ),
        "oracle": "def apply(price, percent):\n    return price * (100 - percent) / 100\n",
        "visible": "from discount_once import apply\n\n\ndef test_visible():\n    assert apply(200, 25) == 150\n",
        "hidden": "from discount_once import apply\n\n\n# orbit-hidden\ndef test_hidden():\n    assert apply(80, 10) == 72\n",
    },
    {
        "id": "slug-spaces",
        "repo": "slug_spaces",
        "entry": "slug_spaces.py",
        "issue": "slugify must lowercase the text and join words with a single hyphen.",
        "buggy": "def slugify(text):\n    return text.lower()\n",
        "oracle": "def slugify(text):\n    return '-'.join(text.lower().split())\n",
        "visible": "from slug_spaces import slugify\n\n\ndef test_visible():\n    assert slugify('Hello World') == 'hello-world'\n",
        "hidden": "from slug_spaces import slugify\n\n\n# orbit-hidden\ndef test_hidden():\n    assert slugify('A  B') == 'a-b'\n",
    },
    {
        "id": "counter-add",
        "repo": "counter_add",
        "entry": "counter_add.py",
        "issue": "Counter.add must increment that key and keep the other keys.",
        "buggy": (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.counts = {}\n"
            "\n"
            "    def add(self, key):\n"
            "        self.counts = {key: 1}\n"
            "\n"
            "    def get(self, key):\n"
            "        return self.counts.get(key, 0)\n"
        ),
        "oracle": (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.counts = {}\n"
            "\n"
            "    def add(self, key):\n"
            "        self.counts[key] = self.counts.get(key, 0) + 1\n"
            "\n"
            "    def get(self, key):\n"
            "        return self.counts.get(key, 0)\n"
        ),
        "visible": (
            "from counter_add import Counter\n\n\n"
            "def test_visible():\n"
            "    counter = Counter()\n"
            "    counter.add('a')\n"
            "    counter.add('a')\n"
            "    assert counter.get('a') == 2\n"
        ),
        "hidden": (
            "from counter_add import Counter\n\n\n"
            "# orbit-hidden\n"
            "def test_hidden():\n"
            "    counter = Counter()\n"
            "    counter.add('a')\n"
            "    counter.add('b')\n"
            "    assert counter.get('a') == 1\n"
            "    assert counter.get('b') == 1\n"
        ),
    },
    {
        "id": "interval-touch",
        "repo": "interval_touch",
        "entry": "interval_touch.py",
        "issue": "overlaps must treat closed intervals as overlapping when they share an endpoint.",
        "buggy": "def overlaps(left, right):\n    return left[0] < right[1] and right[0] < left[1]\n",
        "oracle": "def overlaps(left, right):\n    return left[0] <= right[1] and right[0] <= left[1]\n",
        "visible": "from interval_touch import overlaps\n\n\ndef test_visible():\n    assert overlaps((1, 2), (2, 3)) is True\n",
        "hidden": (
            "from interval_touch import overlaps\n\n\n"
            "# orbit-hidden\n"
            "def test_hidden():\n"
            "    assert overlaps((1, 5), (6, 8)) is False\n"
            "    assert overlaps((0, 1), (1, 1)) is True\n"
        ),
    },
    {
        "id": "median-even",
        "repo": "median_even",
        "entry": "median_even.py",
        "issue": "median of an odd list is the middle value. median of an even list is the average of the two middle values.",
        "buggy": "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n",
        "oracle": (
            "def median(values):\n"
            "    ordered = sorted(values)\n"
            "    mid = len(ordered) // 2\n"
            "    if len(ordered) % 2:\n"
            "        return ordered[mid]\n"
            "    return (ordered[mid - 1] + ordered[mid]) / 2\n"
        ),
        "visible": "from median_even import median\n\n\ndef test_visible():\n    assert median([1, 2, 3, 4]) == 2.5\n",
        "hidden": "from median_even import median\n\n\n# orbit-hidden\ndef test_hidden():\n    assert median([10, 0]) == 5\n    assert median([1, 9, 3]) == 3\n",
    },
    {
        "id": "backoff-pow",
        "repo": "backoff_pow",
        "entry": "backoff_pow.py",
        "issue": "delay(attempt) must be 2 to the power of attempt. attempt 0 is 1.",
        "buggy": "def delay(attempt):\n    return 1\n",
        "oracle": "def delay(attempt):\n    return 2 ** attempt\n",
        "visible": "from backoff_pow import delay\n\n\ndef test_visible():\n    assert delay(3) == 8\n",
        "hidden": "from backoff_pow import delay\n\n\n# orbit-hidden\ndef test_hidden():\n    assert delay(0) == 1\n    assert delay(2) == 4\n",
    },
    {
        "id": "clamp-ends",
        "repo": "clamp_ends",
        "entry": "clamp_ends.py",
        "issue": "clamp(value, low, high) must keep value inside low and high, inclusive.",
        "buggy": (
            "def clamp(value, low, high):\n"
            "    if value > high:\n"
            "        return low\n"
            "    if value < low:\n"
            "        return high\n"
            "    return value\n"
        ),
        "oracle": (
            "def clamp(value, low, high):\n"
            "    if value > high:\n"
            "        return high\n"
            "    if value < low:\n"
            "        return low\n"
            "    return value\n"
        ),
        "visible": "from clamp_ends import clamp\n\n\ndef test_visible():\n    assert clamp(10, 0, 5) == 5\n",
        "hidden": "from clamp_ends import clamp\n\n\n# orbit-hidden\ndef test_hidden():\n    assert clamp(-2, 0, 5) == 0\n    assert clamp(3, 0, 5) == 3\n",
    },
    {
        "id": "stable-unique",
        "repo": "stable_unique",
        "entry": "stable_unique.py",
        "issue": "unique must drop repeats and keep the first occurrence of each item.",
        "buggy": (
            "def unique(items):\n"
            "    out = []\n"
            "    for item in reversed(items):\n"
            "        if item not in out:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        "oracle": (
            "def unique(items):\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if item not in out:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        "visible": "from stable_unique import unique\n\n\ndef test_visible():\n    assert unique([1, 2, 1, 3]) == [1, 2, 3]\n",
        "hidden": "from stable_unique import unique\n\n\n# orbit-hidden\ndef test_hidden():\n    assert unique(['b', 'a', 'a']) == ['b', 'a']\n",
    },
]


def write_cases(root: Path = ROOT) -> None:
    """Write tasks.jsonl and the mini repos. The manifest is written by codebench."""
    scenario = root / "data" / "scenarios" / "code"
    repos = scenario / "repos"
    if repos.exists():
        for child in repos.iterdir():
            if child.is_dir():
                for path in sorted(child.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                child.rmdir()
    lines = []
    for case in CASES:
        repo = repos / case["repo"]
        (repo / "tests").mkdir(parents=True)
        (repo / "hidden").mkdir()
        (repo / "oracle").mkdir()
        (repo / case["entry"]).write_text(case["buggy"], encoding="utf-8")
        (repo / "oracle" / case["entry"]).write_text(case["oracle"], encoding="utf-8")
        (repo / "tests" / "test_visible.py").write_text(case["visible"], encoding="utf-8")
        (repo / "hidden" / "test_hidden.py").write_text(case["hidden"], encoding="utf-8")
        (repo / "ISSUE.txt").write_text(case["issue"] + "\n", encoding="utf-8")
        if "wrong" in case:
            (repo / "wrong").mkdir()
            (repo / "wrong" / case["entry"]).write_text(case["wrong"], encoding="utf-8")
        lines.append(
            {
                "id": case["id"],
                "split": "test",
                "repo": case["repo"],
                "entry": case["entry"],
                "issue": case["issue"],
                "visible_tests": ["tests/test_visible.py"],
                "hidden_tests": ["hidden/test_hidden.py"],
            }
        )
    import json

    body = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in lines)
    (scenario / "tasks.jsonl").write_text(body, encoding="utf-8")


def main() -> None:
    write_cases()
    from orbit_agents.codebench import write_manifest

    path = write_manifest(ROOT)
    print(path)


if __name__ == "__main__":
    main()
