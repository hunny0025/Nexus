"""NEXUS — Intervention outcome simulator for counterfactual reasoning."""

import random


class InterventionSimulator:
    """Simulates the outcome of applying an intervention to the network.

    Each intervention type has different effects on delay, fuel, and
    cascade probability. Outcomes include ±15% random variance.
    """

    def __init__(self):
        self._variance = 0.15

    def _vary(self, value: float) -> float:
        """Apply ±15% variance."""
        return value * random.uniform(1.0 - self._variance, 1.0 + self._variance)

    def run(self, network_state: dict, intervention: dict) -> dict:
        """Simulate the outcome of a single intervention.

        Parameters
        ----------
        network_state : dict
            Current state including: total_delay_minutes, cascade_probability,
            trains_affected (list), fuel_baseline_kg.
        intervention : dict
            Intervention spec from InterventionSpaceGenerator.

        Returns
        -------
        dict
            {total_delay_minutes, fuel_delta_kg, cascade_probability_90min,
             intervention_complexity, trains_contained, trains_still_affected}
        """
        itype = intervention["type"]

        base_delay = network_state.get("total_delay_minutes", 120)
        base_cascade = network_state.get("cascade_probability", 0.8)
        affected = list(network_state.get("trains_affected", []))
        n_affected = len(affected)

        if itype == "REROUTE":
            return self._sim_reroute(base_delay, base_cascade, affected, intervention)
        elif itype == "HOLD":
            return self._sim_hold(base_delay, base_cascade, affected, intervention)
        elif itype == "MAINTENANCE_DISPATCH":
            return self._sim_dispatch(base_delay, base_cascade, affected, intervention)
        elif itype == "COMBINED":
            return self._sim_combined(base_delay, base_cascade, affected, intervention, network_state)
        else:
            # Unknown intervention type — return no change
            return {
                "total_delay_minutes": base_delay,
                "fuel_delta_kg": 0,
                "cascade_probability_90min": base_cascade,
                "intervention_complexity": intervention.get("complexity", 1),
                "trains_contained": 0,
                "trains_still_affected": n_affected,
            }

    def _sim_reroute(self, base_delay, base_cascade, affected, intervention):
        delay_reduction_pct = random.uniform(0.30, 0.60)
        new_delay = self._vary(base_delay * (1 - delay_reduction_pct))
        fuel_delta = self._vary(random.uniform(200, 400))
        new_cascade = self._vary(max(0.0, base_cascade - 0.35))
        contained = 1  # reroute handles one train

        return {
            "total_delay_minutes": round(new_delay, 1),
            "fuel_delta_kg": round(fuel_delta, 1),
            "cascade_probability_90min": round(min(1.0, max(0.0, new_cascade)), 4),
            "intervention_complexity": intervention.get("complexity", 2),
            "trains_contained": contained,
            "trains_still_affected": max(0, len(affected) - contained),
        }

    def _sim_hold(self, base_delay, base_cascade, affected, intervention):
        hold_delay = self._vary(random.uniform(25, 45))
        new_delay = self._vary(base_delay + hold_delay)
        new_cascade = self._vary(max(0.0, base_cascade - 0.20))

        return {
            "total_delay_minutes": round(new_delay, 1),
            "fuel_delta_kg": 0.0,
            "cascade_probability_90min": round(min(1.0, max(0.0, new_cascade)), 4),
            "intervention_complexity": intervention.get("complexity", 1),
            "trains_contained": 1,
            "trains_still_affected": max(0, len(affected) - 1),
        }

    def _sim_dispatch(self, base_delay, base_cascade, affected, intervention):
        # Maintenance takes time but greatly reduces cascade over 60 min
        new_delay = self._vary(base_delay)  # No immediate delay change
        new_cascade = self._vary(max(0.0, base_cascade - 0.50))
        fuel_delta = self._vary(50.0)  # Crew vehicle fuel

        return {
            "total_delay_minutes": round(new_delay, 1),
            "fuel_delta_kg": round(fuel_delta, 1),
            "cascade_probability_90min": round(min(1.0, max(0.0, new_cascade)), 4),
            "intervention_complexity": intervention.get("complexity", 2),
            "trains_contained": 0,
            "trains_still_affected": len(affected),
        }

    def _sim_combined(self, base_delay, base_cascade, affected, intervention, state):
        # Simulate reroute component
        reroute_result = self._sim_reroute(
            base_delay, base_cascade, affected, intervention.get("reroute", {})
        )
        # Simulate dispatch component
        dispatch_result = self._sim_dispatch(
            base_delay, base_cascade, affected, intervention.get("dispatch", {})
        )

        synergy = intervention.get("synergy_bonus", 0.1)

        new_delay = min(reroute_result["total_delay_minutes"], dispatch_result["total_delay_minutes"])
        fuel_delta = reroute_result["fuel_delta_kg"] + dispatch_result["fuel_delta_kg"]
        cascade_reduction = (base_cascade - reroute_result["cascade_probability_90min"]) + \
                           (base_cascade - dispatch_result["cascade_probability_90min"]) + synergy
        new_cascade = max(0.0, base_cascade - cascade_reduction)

        return {
            "total_delay_minutes": round(self._vary(new_delay), 1),
            "fuel_delta_kg": round(fuel_delta, 1),
            "cascade_probability_90min": round(min(1.0, max(0.0, new_cascade)), 4),
            "intervention_complexity": intervention.get("complexity", 3),
            "trains_contained": reroute_result["trains_contained"],
            "trains_still_affected": max(0, len(affected) - reroute_result["trains_contained"]),
        }
