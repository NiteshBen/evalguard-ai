"""Console reporting and rich ANSI terminal dashboard for EvalGuard AI."""

import colorama
from colorama import Fore, Style
from evalguard.models import ClaimStatus, EvalResult, SuiteResult


colorama.init(autoreset=True)


class ConsoleReporter:
    """Renders formatted test suite evaluation summaries and diagnostics to the terminal."""

    @staticmethod
    def print_result(result: EvalResult) -> None:
        """Prints a single test case evaluation summary."""
        tc = result.test_case
        status_badge = (
            f"{Fore.GREEN}[PASS]{Style.RESET_ALL}"
            if result.passed
            else f"{Fore.RED}[FAIL]{Style.RESET_ALL}"
        )

        print("\n" + "=" * 70)
        print(f"Test Case: {Fore.CYAN}{tc.id}{Style.RESET_ALL} -> {status_badge} (Score: {result.overall_score:.2f}, Latency: {result.execution_time_ms}ms)")
        print(f"Query: {tc.query}")
        print(f"Output: {tc.actual_output[:120]}..." if len(tc.actual_output) > 120 else f"Output: {tc.actual_output}")
        print("-" * 70)
        print(f"{'Metric':<22} | {'Score':<8} | {'Threshold':<10} | {'Status':<8} | {'Reason'}")
        print("-" * 70)

        for name, m in result.metrics.items():
            st = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if m.passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            reason_str = (m.reason[:40] + "...") if m.reason and len(m.reason) > 40 else (m.reason or "")
            print(f"{name:<22} | {m.score:<8.2f} | {m.threshold:<10.2f} | {st:<8} | {reason_str}")

        if result.claims:
            print("-" * 70)
            print(f"Claim Verifications ({len(result.claims)} total):")
            for c in result.claims:
                if c.status == ClaimStatus.SUPPORTED:
                    tag = f"{Fore.GREEN}SUPPORTED{Style.RESET_ALL}"
                elif c.status == ClaimStatus.CONTRADICTED:
                    tag = f"{Fore.RED}CONTRADICTED{Style.RESET_ALL}"
                else:
                    tag = f"{Fore.YELLOW}UNVERIFIED{Style.RESET_ALL}"
                print(f"  [{tag}] {c.claim}")
        print("=" * 70)

    @staticmethod
    def print_suite_summary(suite: SuiteResult) -> None:
        """Prints high-level test suite summary table with ANSI colors."""
        sm = suite.summary
        print("\n" + "=" * 75)
        print(f"  {Fore.CYAN}EVALGUARD AI{Style.RESET_ALL} - TEST SUITE RUN: {Fore.YELLOW}{suite.suite_name}{Style.RESET_ALL}")
        print(f"  Timestamp: {suite.timestamp}")
        print("=" * 75)

        pass_color = Fore.GREEN if sm.pass_rate >= 0.8 else (Fore.YELLOW if sm.pass_rate >= 0.5 else Fore.RED)
        print(f"  Total Tests : {sm.total_tests}")
        print(f"  Passed      : {Fore.GREEN}{sm.passed_tests}{Style.RESET_ALL}")
        print(f"  Failed      : {Fore.RED}{sm.failed_tests}{Style.RESET_ALL}")
        print(f"  Pass Rate   : {pass_color}{sm.pass_rate * 100:.1f}%{Style.RESET_ALL}")
        if sm.average_latency_ms is not None:
            print(f"  Avg Latency : {sm.average_latency_ms:.1f} ms")
        if sm.total_tokens is not None:
            print(f"  Total Tokens: {sm.total_tokens}")

        print("\n  Metric Performance Overview:")
        print("  " + "-" * 50)
        print(f"  {'Metric Name':<25} | {'Average Score':<15}")
        print("  " + "-" * 50)
        for name, avg in sm.average_scores.items():
            bar_len = int(avg * 20)
            bar = f"{Fore.GREEN}{'#' * bar_len}{Style.RESET_ALL}{'.' * (20 - bar_len)}"
            print(f"  {name:<25} | {avg:.3f} [{bar}]")

        print("=" * 75 + "\n")
