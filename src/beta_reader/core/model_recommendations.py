"""Model recommendation system for beta-reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ..llm.exceptions import FileProcessingError


@dataclass
class ModelRecommendation:
    """A model recommendation with reasoning."""

    model: str
    score: float
    category: str
    reason: str
    metrics: dict[str, float]


class ModelRecommendationEngine:
    """Recommends optimal models based on performance and quality metrics."""

    def __init__(self) -> None:
        """Initialize recommendation engine."""
        self.console = Console()

    def analyze_results(
        self,
        results: dict[str, dict[str, str | float]],
        priorities: dict[str, float] | None = None,
    ) -> list[ModelRecommendation]:
        """Analyze comparison results and generate recommendations.

        Args:
            results: Results from model comparison.
            priorities: Optional priority weights for different metrics.
                       Keys: 'speed', 'quality', 'consistency', 'efficiency'
                       Values: Weights (0.0-1.0, should sum to 1.0)

        Returns:
            List of model recommendations sorted by score.

        Raises:
            FileProcessingError: If analysis fails.
        """
        if not results:
            raise FileProcessingError("No results to analyze")

        # Default priorities if not provided
        if priorities is None:
            priorities = {
                'speed': 0.3,
                'quality': 0.4,
                'consistency': 0.2,
                'efficiency': 0.1,
            }

        # Validate priorities
        priority_sum = sum(priorities.values())
        if abs(priority_sum - 1.0) > 0.01:
            raise FileProcessingError(f"Priority weights must sum to 1.0, got {priority_sum}")

        successful_results = {
            model: result for model, result in results.items()
            if result.get("success", True)
        }

        if not successful_results:
            raise FileProcessingError("No successful results to analyze")

        recommendations = []

        # Calculate normalized metrics
        speed_scores = self._calculate_speed_scores(successful_results)
        quality_scores = self._calculate_quality_scores(successful_results)
        consistency_scores = self._calculate_consistency_scores(successful_results)
        efficiency_scores = self._calculate_efficiency_scores(successful_results)

        for model, result in successful_results.items():
            # Calculate weighted score
            score = (
                priorities['speed'] * speed_scores.get(model, 0.0) +
                priorities['quality'] * quality_scores.get(model, 0.0) +
                priorities['consistency'] * consistency_scores.get(model, 0.0) +
                priorities['efficiency'] * efficiency_scores.get(model, 0.0)
            )

            # Determine primary category based on strongest metric
            category_scores = {
                'Speed': speed_scores.get(model, 0.0),
                'Quality': quality_scores.get(model, 0.0),
                'Consistency': consistency_scores.get(model, 0.0),
                'Efficiency': efficiency_scores.get(model, 0.0),
            }
            category = max(category_scores, key=category_scores.get)

            # Generate reason
            reason = self._generate_reason(model, result, category_scores)

            recommendation = ModelRecommendation(
                model=model,
                score=score,
                category=category,
                reason=reason,
                metrics={
                    'speed_score': speed_scores.get(model, 0.0),
                    'quality_score': quality_scores.get(model, 0.0),
                    'consistency_score': consistency_scores.get(model, 0.0),
                    'efficiency_score': efficiency_scores.get(model, 0.0),
                    'processing_time': result.get('processing_time', 0.0),
                    'words_per_second': result.get('words_per_second', 0.0),
                    'word_count': result.get('word_count', 0),
                }
            )
            recommendations.append(recommendation)

        # Sort by score (descending)
        return sorted(recommendations, key=lambda r: r.score, reverse=True)

    def _calculate_speed_scores(self, results: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate normalized speed scores (0.0-1.0)."""
        wps_values = [r.get('words_per_second', 0.0) for r in results.values()]
        max_wps = max(wps_values) if wps_values else 1.0

        if max_wps == 0:
            return dict.fromkeys(results, 0.0)

        return {
            model: result.get('words_per_second', 0.0) / max_wps
            for model, result in results.items()
        }

    def _calculate_quality_scores(self, results: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate quality scores based on output length and completeness."""
        # Use word count as a proxy for quality (more thorough editing)
        word_counts = [r.get('word_count', 0) for r in results.values()]
        max_words = max(word_counts) if word_counts else 1
        min_words = min(word_counts) if word_counts else 0

        if max_words == min_words:
            return dict.fromkeys(results, 1.0)

        # Normalize word counts (higher is better for editing quality)
        return {
            model: (result.get('word_count', 0) - min_words) / (max_words - min_words)
            for model, result in results.items()
        }

    def _calculate_consistency_scores(self, results: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate consistency scores based on output variability."""
        # For now, use inverse of processing time variance as proxy
        # In future, this could compare multiple runs or analyze output patterns
        times = [r.get('processing_time', 0.0) for r in results.values()]
        avg_time = sum(times) / len(times) if times else 1.0

        scores = {}
        for model, result in results.items():
            time = result.get('processing_time', 0.0)
            # Lower variance from average = higher consistency
            variance = abs(time - avg_time) / avg_time if avg_time > 0 else 0
            scores[model] = max(0.0, 1.0 - variance)

        return scores

    def _calculate_efficiency_scores(self, results: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate efficiency scores (output quality per time unit)."""
        efficiency_values = []
        for result in results.values():
            words = result.get('word_count', 0)
            time = result.get('processing_time', 1.0)
            if time > 0:
                efficiency_values.append(words / time)
            else:
                efficiency_values.append(0.0)

        max_efficiency = max(efficiency_values) if efficiency_values else 1.0

        if max_efficiency == 0:
            return dict.fromkeys(results, 0.0)

        scores = {}
        for model, result in results.items():
            words = result.get('word_count', 0)
            time = result.get('processing_time', 1.0)
            efficiency = (words / time) if time > 0 else 0.0
            scores[model] = efficiency / max_efficiency

        return scores

    def _generate_reason(
        self,
        model: str,
        result: dict[str, Any],
        category_scores: dict[str, float]
    ) -> str:
        """Generate human-readable reason for recommendation."""
        max_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[max_category]

        time = result.get('processing_time', 0.0)
        wps = result.get('words_per_second', 0.0)
        words = result.get('word_count', 0)

        if max_category == 'Speed':
            return f"Fastest processing at {wps:.1f} words/second ({time:.1f}s total)"
        elif max_category == 'Quality':
            return f"Highest quality output with {words:,} words generated"
        elif max_category == 'Consistency':
            return "Most consistent performance with stable processing time"
        else:  # Efficiency
            return f"Best balance of speed and quality ({wps:.1f} words/sec, {words:,} words)"

    def display_recommendations(
        self,
        recommendations: list[ModelRecommendation],
        show_metrics: bool = False
    ) -> None:
        """Display recommendations in a formatted table.

        Args:
            recommendations: List of model recommendations.
            show_metrics: Whether to show detailed metrics.
        """
        if not recommendations:
            self.console.print("[yellow]No recommendations to display[/yellow]")
            return

        # Main recommendations table
        table = Table(title="Model Recommendations")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Model", style="green")
        table.add_column("Category", style="blue")
        table.add_column("Score", style="yellow", justify="right")
        table.add_column("Reason", style="white")

        for i, rec in enumerate(recommendations, 1):
            rank = f"#{i}"
            if i == 1:
                rank = f"[bold green]{rank}[/bold green]"
            elif i <= 3:
                rank = f"[green]{rank}[/green]"

            score_str = f"{rec.score:.3f}"
            if i == 1:
                score_str = f"[bold]{score_str}[/bold]"

            table.add_row(
                rank,
                rec.model,
                rec.category,
                score_str,
                rec.reason
            )

        self.console.print(table)

        # Show detailed metrics if requested
        if show_metrics:
            self.console.print("\n[bold]Detailed Metrics:[/bold]")
            metrics_table = Table()
            metrics_table.add_column("Model", style="cyan")
            metrics_table.add_column("Speed", style="green", justify="right")
            metrics_table.add_column("Quality", style="blue", justify="right")
            metrics_table.add_column("Consistency", style="yellow", justify="right")
            metrics_table.add_column("Efficiency", style="magenta", justify="right")

            for rec in recommendations:
                metrics_table.add_row(
                    rec.model,
                    f"{rec.metrics['speed_score']:.3f}",
                    f"{rec.metrics['quality_score']:.3f}",
                    f"{rec.metrics['consistency_score']:.3f}",
                    f"{rec.metrics['efficiency_score']:.3f}",
                )

            self.console.print(metrics_table)

    def save_recommendations_report(
        self,
        recommendations: list[ModelRecommendation],
        output_path: Path,
        priorities: dict[str, float] | None = None,
    ) -> None:
        """Save recommendations report to file.

        Args:
            recommendations: List of recommendations to save.
            output_path: Path to save the report.
            priorities: Priority weights used for scoring.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Recommendations Report\n\n")

            f.write(f"**Generated on:** {Path(__file__).stat().st_mtime}\n")
            f.write(f"**Models analyzed:** {len(recommendations)}\n\n")

            if priorities:
                f.write("## Scoring Priorities\n\n")
                f.write("| Metric | Weight |\n")
                f.write("|--------|--------|\n")
                for metric, weight in priorities.items():
                    f.write(f"| {metric.title()} | {weight:.1%} |\n")
                f.write("\n")

            # Recommendations table
            f.write("## Recommendations\n\n")
            f.write("| Rank | Model | Category | Score | Reason |\n")
            f.write("|------|-------|----------|-------|--------|\n")

            for i, rec in enumerate(recommendations, 1):
                f.write(f"| #{i} | {rec.model} | {rec.category} | "
                       f"{rec.score:.3f} | {rec.reason} |\n")

            f.write("\n")

            # Detailed metrics
            f.write("## Detailed Metrics\n\n")
            f.write("| Model | Speed | Quality | Consistency | Efficiency | WPS | Time(s) |\n")
            f.write("|-------|-------|---------|-------------|------------|-----|--------|\n")

            for rec in recommendations:
                f.write(f"| {rec.model} | {rec.metrics['speed_score']:.3f} | "
                       f"{rec.metrics['quality_score']:.3f} | "
                       f"{rec.metrics['consistency_score']:.3f} | "
                       f"{rec.metrics['efficiency_score']:.3f} | "
                       f"{rec.metrics['words_per_second']:.1f} | "
                       f"{rec.metrics['processing_time']:.2f} |\n")

            f.write("\n")

            # Top recommendation details
            if recommendations:
                top_rec = recommendations[0]
                f.write("## Top Recommendation\n\n")
                f.write(f"**Model:** {top_rec.model}\n")
                f.write(f"**Category:** {top_rec.category}\n")
                f.write(f"**Score:** {top_rec.score:.3f}\n")
                f.write(f"**Reason:** {top_rec.reason}\n\n")

                f.write("**Performance Details:**\n")
                f.write(f"- Processing time: {top_rec.metrics['processing_time']:.2f} seconds\n")
                f.write(f"- Words per second: {top_rec.metrics['words_per_second']:.1f}\n")
                f.write(f"- Words generated: {top_rec.metrics['word_count']:,}\n")

        self.console.print(f"[green]Recommendations report saved to: {output_path}[/green]")

    def get_recommendation_for_use_case(
        self,
        recommendations: list[ModelRecommendation],
        use_case: str,
    ) -> ModelRecommendation | None:
        """Get the best recommendation for a specific use case.

        Args:
            recommendations: List of available recommendations.
            use_case: Use case ('speed', 'quality', 'balanced', 'efficiency').

        Returns:
            Best recommendation for the use case, or None if no match.
        """
        if not recommendations:
            return None

        if use_case.lower() == 'speed':
            # Find model with highest speed score
            return max(recommendations, key=lambda r: r.metrics['speed_score'])
        elif use_case.lower() == 'quality':
            # Find model with highest quality score
            return max(recommendations, key=lambda r: r.metrics['quality_score'])
        elif use_case.lower() == 'efficiency':
            # Find model with highest efficiency score
            return max(recommendations, key=lambda r: r.metrics['efficiency_score'])
        else:  # balanced or default
            # Return highest overall score
            return recommendations[0] if recommendations else None
