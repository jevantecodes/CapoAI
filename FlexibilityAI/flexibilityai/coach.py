from __future__ import annotations

import os
from typing import Any


class CoachingResponder:
    def __init__(self, *, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(
        self,
        *,
        athlete: dict[str, Any],
        analytics: dict[str, Any],
        goal: str | None,
    ) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                message = self._generate_with_openai(
                    api_key=api_key,
                    athlete=athlete,
                    analytics=analytics,
                    goal=goal,
                )
                return {
                    "provider": "openai",
                    "model": self.model,
                    "message": message,
                }
            except Exception as exc:
                fallback = self._generate_heuristic(athlete=athlete, analytics=analytics, goal=goal)
                return {
                    "provider": "heuristic_fallback",
                    "model": self.model,
                    "error": str(exc),
                    "message": fallback,
                }

        fallback = self._generate_heuristic(athlete=athlete, analytics=analytics, goal=goal)
        return {
            "provider": "heuristic",
            "model": None,
            "message": fallback,
        }

    def _generate_with_openai(
        self,
        *,
        api_key: str,
        athlete: dict[str, Any],
        analytics: dict[str, Any],
        goal: str | None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        target_goal = goal or "their chosen capoeira movement"
        system_prompt = (
            "You are a flexibility and capoeira prep coach. "
            "Give practical, safe, concise guidance based on athlete analytics. "
            "Use plain language and include measurable next steps for 2 weeks."
        )
        user_prompt = (
            f"Athlete profile: {athlete}\n"
            f"Analytics: {analytics}\n"
            f"Target goal: {target_goal}\n"
            "Return exactly three short sections titled: Current Status, Priority Fixes, 2-Week Plan."
        )

        response = client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            raise RuntimeError("OpenAI returned an empty coaching response.")
        return content

    def _generate_heuristic(self, *, athlete: dict[str, Any], analytics: dict[str, Any], goal: str | None) -> str:
        name = athlete.get("full_name") or "Athlete"
        quality = analytics.get("quality", {})
        readiness = analytics.get("readiness", {})
        movement_breakdown = analytics.get("movement_breakdown", {})

        top_needs = sorted(
            movement_breakdown.items(),
            key=lambda kv: float(kv[1].get("latest_quality", 0.0)),
        )[:2]

        priorities = [movement for movement, _ in top_needs] or ["bridge", "lunge_stretch"]
        goal_text = goal or "your target capoeira movement"

        avg_quality = quality.get("avg")
        latest_quality = quality.get("latest")
        quality_change = quality.get("improvement_recent_vs_baseline")
        latest_readiness = readiness.get("latest")

        lines = [
            f"Current Status: {name}, your latest quality score is {latest_quality} and average quality is {avg_quality}. "
            f"Recent trend change is {quality_change} points. Goal readiness for {goal_text} is currently {latest_readiness}.",
            f"Priority Fixes: Focus first on {priorities[0]} and {priorities[1]}. "
            "Aim for controlled reps with full range and stable posture before speed.",
            "2-Week Plan: 4 sessions/week. In each session do 3 rounds: "
            "(1) mobility prep 6-8 min, (2) movement holds 3x30-45s, (3) technique reps 3x8 slow reps. "
            "Retest quality scores every 3-4 days and target +6 to +10 points in your weakest movement.",
        ]
        return "\n\n".join(lines)
