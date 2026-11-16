import math
import os
import random


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class SNNDecision:
    """
    Potential-field scorer — v2 fixes applied:

    F4  temperature ladder: 1.8 (far) → 0.5 (mid) → 0.05 (near) → greedy (final)
        Restores decision diversity that temperature=0.3 was killing.
    F2  w_pheromone=8.0 added so shared trail actually influences decisions.
        (Feature is now fed in from build_features via feat['pheromone'].)
    Tuning:
        w_progress reduced 15→9 to stop greedy tunneling into dead-end corridors.
        w_explore raised 0.5→2.0 to widen the search cone.
        exploration_rate raised 0.05→0.12.
    """

    def __init__(
        self,
        rng=None,
        temperature_far=1.8,
        temperature_mid=0.5,
        temperature_near=0.05,
        exploration_rate=0.12,
        greedy_distance_m=50.0,
        near_distance_m=200.0,
        mid_distance_m=800.0,
    ):
        # Own random stream (a random.Random). Falls back to the global module
        # so existing scripts behave exactly as before.
        self.rng = rng if rng is not None else random
        self.w_goal       = 10.0
        self.w_hazard     = -8.0
        self.w_visited    = -1.5   # applied as log-penalty now, see score_neighbor
        self.w_backtrack  = -6.0
        self.w_progress   = 9.0    # FIX: was 15.0 — greedy tunneling
        self.w_explore    = 2.0    # FIX: was 0.5
        self.w_taboo      = -200.0
        self.w_blacklist  = -500.0
        self.w_two_hop    = 30.0
        self.w_momentum   = 4.0
        # Default remains 8.0. For controlled sensitivity experiments, set
        # SWARM_PHEROMONE_WEIGHT in the environment before launching Python.
        # This changes only the pheromone score coefficient; all other policy
        # terms and decision logic remain unchanged.
        configured_pheromone_weight = os.getenv("SWARM_PHEROMONE_WEIGHT")
        self.w_pheromone = (
            8.0 if configured_pheromone_weight is None
            else float(configured_pheromone_weight)
        )
        if not math.isfinite(self.w_pheromone) or self.w_pheromone < 0.0:
            raise ValueError("SWARM_PHEROMONE_WEIGHT must be a finite non-negative number")
        # Knowledge-driven terms (fed by what the agent has sensed or been told)
        self.w_lookahead  = -6.0   # mean risk of the candidate's onward exits (known blocked=1)
        self.w_dead_end   = -25.0  # every onward exit of the candidate is known blocked
        self.w_plan       = 12.0   # candidate is the next hop of the agent's own plan

        self.momentum_decay   = 0.7
        self.last_direction   = None
        # Selection policy parameters are configurable so experiments can
        # inspect/tune them without editing the decision logic. Defaults
        # preserve the existing policy exactly.
        self.temperature_far = float(temperature_far)
        self.temperature_mid = float(temperature_mid)
        self.temperature_near = float(temperature_near)
        self.exploration_rate = float(exploration_rate)
        self.greedy_distance_m = float(greedy_distance_m)
        self.near_distance_m = float(near_distance_m)
        self.mid_distance_m = float(mid_distance_m)

        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between 0 and 1")
        if min(
            self.temperature_far, self.temperature_mid,
            self.temperature_near
        ) <= 0:
            raise ValueError("Softmax temperatures must be > 0")
        if not (
            0 <= self.greedy_distance_m <= self.near_distance_m
            <= self.mid_distance_m
        ):
            raise ValueError(
                "Distance thresholds must satisfy "
                "0 <= greedy <= near <= mid"
            )

        # Latest choice diagnostics. Read-only-by-convention snapshot,
        # overwritten on each call to decide().
        self.last_decision = None

    def reset_momentum(self):
        self.last_direction = None

    def apply_weights(self, overrides: dict):
        """Override scorer parameters by name (used by role profiles).
        Unknown names raise, so a typo in a role spec cannot silently do nothing."""
        for k, v in (overrides or {}).items():
            if not hasattr(self, k):
                raise AttributeError(f"SNNDecision has no parameter {k!r}")
            setattr(self, k, v)

    def compute_momentum_score(self, current_pos, next_pos):
        if self.last_direction is None:
            return 0.0
        dlat = next_pos['lat'] - current_pos['lat']
        dlon = next_pos['lon'] - current_pos['lon']
        mag  = math.sqrt(dlat**2 + dlon**2)
        if mag < 1e-9:
            return 0.0
        dot = (dlat / mag) * self.last_direction[0] + (dlon / mag) * self.last_direction[1]
        return dot * self.w_momentum

    def update_momentum(self, current_pos, next_pos):
        dlat = next_pos['lat'] - current_pos['lat']
        dlon = next_pos['lon'] - current_pos['lon']
        mag  = math.sqrt(dlat**2 + dlon**2)
        if mag < 1e-9:
            return
        nd = (dlat / mag, dlon / mag)
        if self.last_direction is None:
            self.last_direction = nd
        else:
            self.last_direction = (
                self.momentum_decay * self.last_direction[0] + (1 - self.momentum_decay) * nd[0],
                self.momentum_decay * self.last_direction[1] + (1 - self.momentum_decay) * nd[1],
            )

    def score_neighbor(self, feat, current_pos, goal_pos):
        d        = feat['dist']
        in_final = d < 80
        in_near  = d < 200
        in_mid   = d < 800

        # 1. Goal attraction
        dist_norm = min(d, 5000.0) / 5000.0
        base_goal = (1.0 - dist_norm) * self.w_goal
        if in_final:
            goal_score = base_goal * (1.0 + (80 - d) / 80 * 8.0)
        elif in_near:
            goal_score = base_goal * 2.5
        else:
            goal_score = base_goal

        # 2. Progress (reduced weight to avoid greedy tunneling — FIX)
        progress_score = feat.get('progress', 0.0) * self.w_progress

        # 3. Hazard
        haz_mult     = 0.1 if in_final else (0.5 if in_near else 1.0)
        hazard_score = feat['hazard'] * self.w_hazard * haz_mult

        # 4. Visited — soft log penalty so repeat visits compound (FIX: was binary 0/1/2)
        raw_visit    = feat.get('visited', 0.0)          # now a float from log1p
        vis_mult     = 0.3 if in_near else 1.0
        visited_score = raw_visit * self.w_visited * vis_mult

        # 5. Backtrack
        bt_mult        = 0.4 if in_near else 1.0
        backtrack_score = feat.get('is_prev_node', 0.0) * self.w_backtrack * bt_mult

        # 6. Exploration (suppressed near goal)
        explore_score = 0.0 if in_near else feat.get('is_new', 0.0) * self.w_explore

        # 7. Momentum
        mom_score = 0.0
        if 'node_data' in feat and current_pos is not None:
            mom_score = self.compute_momentum_score(current_pos, feat['node_data'])
            if in_near:
                mom_score *= 0.5

        # 8. Taboo
        taboo_score = feat.get('taboo', 0.0) * self.w_taboo

        # 9. Blacklist (near-infinite)
        bl_score = feat.get('blacklist', 0.0) * self.w_blacklist

        # 10. Two-hop goal bonus
        two_hop_score = feat.get('two_hop_goal', 0.0) * self.w_two_hop

        # 11. Pheromone trail bonus — FIX F2: was never scored
        pheromone_score = feat.get('pheromone', 0.0) * self.w_pheromone

        # 12. Knowledge terms. Dead-end avoidance is dropped in the final approach
        #     where the goal itself is the exit.
        look_score = feat.get('lookahead', 0.0) * self.w_lookahead
        if not in_final:
            look_score += feat.get('dead_end', 0.0) * self.w_dead_end
        plan_score = feat.get('on_plan', 0.0) * self.w_plan

        return (goal_score + progress_score + hazard_score + visited_score +
                backtrack_score + explore_score + mom_score +
                taboo_score + bl_score + two_hop_score + pheromone_score +
                look_score + plan_score)

    def softmax_selection(self, scores, temperature, return_probs=False):
        """Sample a score-proportional choice.

        The default return remains the selected index for backward
        compatibility. Set return_probs=True to also receive the probability
        vector for scientific diagnostics.
        """
        if not scores:
            if return_probs:
                return None, []
            return None
        if temperature is None or temperature <= 0:
            raise ValueError("temperature must be > 0")

        max_s = max(scores)
        exp_s = [math.exp((float(s) - max_s) / temperature) for s in scores]
        total = sum(exp_s)
        probs = [value / total for value in exp_s]

        r = self.rng.random()
        cumulative = 0.0
        selected = len(scores) - 1
        for i, probability in enumerate(probs):
            cumulative += probability
            if r <= cumulative:
                selected = i
                break

        if return_probs:
            return selected, probs
        return selected

    @staticmethod
    def _probability_entropy(probabilities):
        """Shannon entropy in nats; zero for a deterministic distribution."""
        return -sum(
            p * math.log(p) for p in probabilities if p > 0.0
        )

    @staticmethod
    def _deduplicate_features(neighbor_features):
        """Keep exactly one feature record per destination node.

        Adjacency lists may contain repeated entries (for example, parallel
        road edges). Navigation is a node-choice problem here, so parallel
        edges must not multiply a destination's probability mass.
        """
        unique = []
        seen = set()
        duplicate_count = 0
        for feature in neighbor_features:
            node = feature.get("node")
            if node in seen:
                duplicate_count += 1
                continue
            seen.add(node)
            unique.append(feature)
        return unique, duplicate_count

    @staticmethod
    def _softmax_probabilities(scores, temperature):
        """Return a numerically stable softmax probability vector."""
        if not scores:
            return []
        if temperature is None or temperature <= 0:
            raise ValueError("temperature must be > 0")
        max_score = max(float(score) for score in scores)
        weights = [
            math.exp((float(score) - max_score) / temperature)
            for score in scores
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    def decide(self, neighbor_features, current_pos=None, goal_pos=None, goal_node=None):
        raw_candidate_count = len(neighbor_features or [])
        neighbor_features, duplicate_candidate_count = (
            self._deduplicate_features(neighbor_features or [])
        )

        if not neighbor_features:
            self.last_decision = {
                "selection_method": "no_candidates",
                "candidate_count": 0,
                "raw_candidate_count": raw_candidate_count,
                "duplicate_candidate_count": duplicate_candidate_count,
                "selected_node": None,
                "candidates": [],
            }
            return None

        # Always grab goal if adjacent. This is an intentional deterministic
        # rule and is recorded separately from score-based selection.
        if goal_node is not None:
            for feature in neighbor_features:
                if feature.get("node") == goal_node:
                    self.last_decision = {
                        "selection_method": "goal_adjacent",
                        "candidate_count": len(neighbor_features),
                        "raw_candidate_count": raw_candidate_count,
                        "duplicate_candidate_count": duplicate_candidate_count,
                        "selected_node": goal_node,
                        "candidates": [],
                        "goal_adjacent_override": True,
                    }
                    return goal_node

        dist_to_goal = float("inf")
        if goal_pos is not None and current_pos is not None:
            dist_to_goal = haversine_distance(
                current_pos["lat"], current_pos["lon"],
                goal_pos["lat"], goal_pos["lon"]
            )

        scores = [
            self.score_neighbor(feature, current_pos, goal_pos)
            for feature in neighbor_features
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)

        if dist_to_goal < self.greedy_distance_m:
            selected_index = best_index
            probabilities = [0.0] * len(scores)
            probabilities[best_index] = 1.0
            selection_method = "greedy"
            exploration_used = False
            temperature = None
        elif dist_to_goal < self.near_distance_m:
            temperature = self.temperature_near
            selected_index, probabilities = self.softmax_selection(
                scores, temperature, return_probs=True
            )
            selection_method = "softmax_near"
            exploration_used = False
        elif dist_to_goal < self.mid_distance_m:
            temperature = self.temperature_mid
            selected_index, probabilities = self.softmax_selection(
                scores, temperature, return_probs=True
            )
            selection_method = "softmax_mid"
            exploration_used = False
        else:
            temperature = self.temperature_far
            # Exploration samples only from already legal candidates supplied
            # by SNNAgent; it cannot create or inject an unsafe move.
            if self.rng.random() < self.exploration_rate:
                unvisited_indices = [
                    i for i, feature in enumerate(neighbor_features)
                    if feature.get("is_new", 0.0) > 0.5
                ]
                pool = unvisited_indices or list(range(len(neighbor_features)))
                selected_index = self.rng.choice(pool)
                probabilities = [0.0] * len(scores)
                pool_probability = 1.0 / len(pool)
                for i in pool:
                    probabilities[i] = pool_probability
                selection_method = "uniform_exploration"
                exploration_used = True
            else:
                selected_index, probabilities = self.softmax_selection(
                    scores, temperature, return_probs=True
                )
                selection_method = "softmax_far"
                exploration_used = False

        # Counterfactual: hold the candidate set and temperature fixed, but
        # remove only the pheromone term. This isolates the direct score effect.
        pheromone_scores = [
            float(feature.get("pheromone", 0.0)) * self.w_pheromone
            for feature in neighbor_features
        ]
        scores_without_pheromone = [
            score - pheromone_score
            for score, pheromone_score in zip(scores, pheromone_scores)
        ]

        if selection_method.startswith("softmax_"):
            counterfactual_probabilities = self._softmax_probabilities(
                scores_without_pheromone, temperature
            )
            counterfactual_method = "softmax_same_temperature_without_pheromone"
        elif selection_method == "greedy":
            cf_best = max(
                range(len(scores_without_pheromone)),
                key=scores_without_pheromone.__getitem__
            )
            counterfactual_probabilities = [0.0] * len(scores)
            counterfactual_probabilities[cf_best] = 1.0
            counterfactual_method = "greedy_without_pheromone"
        else:
            # Uniform exploration is not score-based; a softmax counterfactual
            # would not describe the actual exploration mechanism.
            counterfactual_probabilities = [None] * len(scores)
            counterfactual_method = "not_applicable_uniform_exploration"

        max_score = max(scores)
        min_score = min(scores)
        sorted_scores = sorted(scores, reverse=True)
        probability_entropy = self._probability_entropy(probabilities)
        selected_feature = neighbor_features[selected_index]
        candidates = []
        for index, (feature, score) in enumerate(zip(neighbor_features, scores)):
            cf_probability = counterfactual_probabilities[index]
            candidates.append({
                "node": feature.get("node"),
                "score": score,
                "probability": probabilities[index],
                "pheromone_feature": feature.get("pheromone", 0.0),
                "pheromone_score": pheromone_scores[index],
                "score_without_pheromone": scores_without_pheromone[index],
                "probability_without_pheromone": cf_probability,
                "pheromone_probability_effect": (
                    probabilities[index] - cf_probability
                    if cf_probability is not None else None
                ),
                "is_new": feature.get("is_new", 0.0),
            })

        cf_best_index = max(
            range(len(scores_without_pheromone)),
            key=scores_without_pheromone.__getitem__
        )
        self.last_decision = {
            "selection_method": selection_method,
            "distance_to_goal_m": (
                dist_to_goal if math.isfinite(dist_to_goal) else None
            ),
            "candidate_count": len(neighbor_features),
            "raw_candidate_count": raw_candidate_count,
            "duplicate_candidate_count": duplicate_candidate_count,
            "selected_index": selected_index,
            "selected_node": selected_feature.get("node"),
            "best_index": best_index,
            "best_node": neighbor_features[best_index].get("node"),
            "selected_is_best_scoring": selected_index == best_index,
            "selected_probability": probabilities[selected_index],
            "best_candidate_probability": probabilities[best_index],
            "score_gap_best_second": (
                max_score - sorted_scores[1] if len(sorted_scores) > 1 else None
            ),
            "score_range": max_score - min_score,
            "probability_entropy_nats": probability_entropy,
            "exploration_used": exploration_used,
            "temperature": temperature,
            "pheromone_weight": self.w_pheromone,
            "exploration_rate": self.exploration_rate,
            "counterfactual_method": counterfactual_method,
            "best_node_without_pheromone": neighbor_features[cf_best_index].get("node"),
            "pheromone_changed_score_argmax": (
                neighbor_features[best_index].get("node")
                != neighbor_features[cf_best_index].get("node")
            ),
            "candidates": candidates,
        }
        return selected_feature["node"]
