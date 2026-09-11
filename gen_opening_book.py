#!/usr/bin/env python3
"""Precompute first-hand advice so the helper can answer instantly."""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from multiprocessing import cpu_count

COLORS = ("blue", "red", "yellow")
COLOR_PERMS = (
    ("blue", "red", "yellow"),
    ("blue", "yellow", "red"),
    ("red", "blue", "yellow"),
    ("red", "yellow", "blue"),
    ("yellow", "blue", "red"),
    ("yellow", "red", "blue"),
)
SILVER = 300
GOLD = 400
COLOR_OF = [i // 8 for i in range(24)]
RANK_OF = [i % 8 + 1 for i in range(24)]
KEY_OF = [f"{COLORS[i // 8]}-{i % 8 + 1}" for i in range(24)]
BIT_OF = [1 << i for i in range(24)]
KEY_INDEX = {KEY_OF[i]: i for i in range(24)}


def group_score(n):
    return 10 * n + 10


def seq_score(mn, same):
    return (40 if same else 0) + 10 * mn


def to_i32(n):
    n &= 0xFFFFFFFF
    return n - 0x100000000 if n >= 0x80000000 else n


def hash32(text):
    h = to_i32(2166136261)
    for ch in text:
        h = to_i32(h ^ ord(ch))
        h = to_i32(h * 16777619)
    return h & 0xFFFFFFFF


def mix32(value):
    x = value & 0xFFFFFFFF
    x ^= x >> 16
    x = to_i32(to_i32(x) * 2246822507) & 0xFFFFFFFF
    x ^= x >> 13
    x = to_i32(to_i32(x) * 3266489909) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def ordered_deck(card_ids, base, index):
    if not card_ids:
        return []
    canonical = sorted(card_ids, key=lambda i: KEY_OF[i])
    first = canonical[index % len(canonical)]
    seed = mix32(base ^ ((index + 1) * 2654435761 & 0xFFFFFFFF))
    rest = [i for i in canonical if i != first]
    rest.sort(key=lambda i: (mix32(seed ^ hash32(KEY_OF[i])), KEY_OF[i]))
    rest.append(first)
    return rest


def make_rollout_seeds(n, state_key):
    base = hash32(state_key)
    return [(base, i) for i in range(n)]


def classify_ids(a, b, c):
    ranks = sorted((RANK_OF[a], RANK_OF[b], RANK_OF[c]))
    colors = (COLOR_OF[a], COLOR_OF[b], COLOR_OF[c])
    if ranks[0] == ranks[1] == ranks[2]:
        if len(set(colors)) != 3:
            return None
        return (group_score(ranks[0]), 1)
    if ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1:
        same = colors[0] == colors[1] == colors[2]
        return (seq_score(ranks[0], same), 2 if same else 4)
    return None


COMBO_INFO = {}
GOOD_COMBOS = []
for a, b, c in combinations(range(24), 3):
    info = classify_ids(a, b, c)
    if not info:
        continue
    mask = BIT_OF[a] | BIT_OF[b] | BIT_OF[c]
    COMBO_INFO[mask] = (info[0], info[1], (a, b, c))
    if info[1] in (1, 2):
        GOOD_COMBOS.append((info[0], mask))
GOOD_COMBOS.sort(reverse=True)


def find_combos(hand):
    out = []
    n = len(hand)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = BIT_OF[hand[i]] | BIT_OF[hand[j]] | BIT_OF[hand[k]]
                info = COMBO_INFO.get(mask)
                if info:
                    out.append((info[0], info[1], mask, info[2]))
    return out


def has_card(mask, color_i, rank):
    return bool(mask & BIT_OF[color_i * 8 + rank - 1])


def keep_value(card, pool_mask):
    best = 0
    rank = RANK_OF[card]
    color_i = COLOR_OF[card]
    for start in (rank - 2, rank - 1, rank):
        if start < 1 or start > 6:
            continue
        if all(has_card(pool_mask, color_i, r) for r in (start, start + 1, start + 2)):
            best = max(best, seq_score(start, True))
    if all(has_card(pool_mask, col, rank) for col in range(3)):
        best = max(best, group_score(rank))
    return best


def combo_destroys(ids, pool_mask):
    return any(keep_value(card, pool_mask) >= 50 for card in ids)


OPT_CACHE = {}
MAX_CACHE = {}


def optimistic_good(mask):
    cached = OPT_CACHE.get(mask)
    if cached is not None:
        return cached
    pts = 0
    left = mask
    while True:
        best_s = 0
        best_m = 0
        for score, cm in GOOD_COMBOS:
            if left & cm == cm and score > best_s:
                best_s = score
                best_m = cm
        if not best_s:
            OPT_CACHE[mask] = pts
            return pts
        pts += best_s
        left ^= best_m


def cards_from_mask(mask):
    return [i for i in range(24) if mask & BIT_OF[i]]


def max_points_exact(card_ids):
    n = len(card_ids)
    if n < 3:
        return 0
    combos = []
    for i, j, k in combinations(range(n), 3):
        info = COMBO_INFO.get(BIT_OF[card_ids[i]] | BIT_OF[card_ids[j]] | BIT_OF[card_ids[k]])
        if info:
            combos.append(((1 << i) | (1 << j) | (1 << k), info[0]))
    if not combos:
        return 0
    memo = [-1] * (1 << n)

    def rec(cur):
        cached = memo[cur]
        if cached >= 0:
            return cached
        best = 0
        for bits, score in combos:
            if cur & bits == bits:
                v = score + rec(cur ^ bits)
                if v > best:
                    best = v
        memo[cur] = best
        return best

    return rec((1 << n) - 1)


def max_points(mask):
    n = mask.bit_count()
    if n < 3:
        return 0
    cached = MAX_CACHE.get(mask)
    if cached is not None:
        return cached
    if n <= 16:
        value = max_points_exact(cards_from_mask(mask))
    else:
        value = optimistic_good(mask)
        left = mask
        pts = 0
        while True:
            best_s = 0
            best_m = 0
            for score, cm in GOOD_COMBOS:
                if left & cm == cm and score > best_s:
                    best_s = score
                    best_m = cm
            if not best_s:
                break
            pts += best_s
            left ^= best_m
        value = pts + max_points_exact(cards_from_mask(left))
    MAX_CACHE[mask] = value
    return value


def color_run_needs(hand4, deck_mask):
    outs = []
    n = len(hand4)
    for i in range(n):
        a = hand4[i]
        for j in range(i + 1, n):
            b = hand4[j]
            if COLOR_OF[a] != COLOR_OF[b]:
                continue
            color_i = COLOR_OF[a]
            ra, rb = RANK_OF[a], RANK_OF[b]
            gap = abs(ra - rb)
            needs = []
            if gap == 1:
                lo, hi = min(ra, rb), max(ra, rb)
                if lo > 1:
                    needs.append((color_i, lo - 1, seq_score(lo - 1, True)))
                if hi < 8:
                    needs.append((color_i, hi + 1, seq_score(lo, True)))
            elif gap == 2:
                lo = min(ra, rb)
                needs.append((color_i, (ra + rb) // 2, seq_score(lo, True)))
            for color_i, rank, score in needs:
                if has_card(deck_mask, color_i, rank):
                    outs.append((color_i, rank, score))
    return outs


def pair_value(needs):
    seen = set()
    total = 0
    for item in needs:
        if item in seen:
            continue
        seen.add(item)
        total += item[2]
    return total


def seeded_combo_value(hand, pool_mask):
    total = 0
    hand_set = set(hand)
    for color_i in range(3):
        seqs = []
        for start in range(1, 7):
            ids = [color_i * 8 + start - 1 + off for off in range(3)]
            if not all(pool_mask & BIT_OF[i] for i in ids):
                continue
            seeds = sum(1 for i in ids if i in hand_set)
            if not seeds:
                continue
            sc = seq_score(start, True)
            total += sc if seeds >= 2 else int(sc * 0.5 + 0.5)
            seqs.append((ids, sc))
        two = 0
        for i in range(len(seqs)):
            set_i = set(seqs[i][0])
            for j in range(i + 1, len(seqs)):
                if set_i.isdisjoint(seqs[j][0]):
                    two = max(two, min(seqs[i][1], seqs[j][1]))
        total += two
    for n in range(1, 9):
        ids = [col * 8 + n - 1 for col in range(3)]
        if not all(pool_mask & BIT_OF[i] for i in ids):
            continue
        seeds = sum(1 for i in ids if i in hand_set)
        if not seeds:
            continue
        sc = group_score(n)
        total += sc if seeds >= 2 else int(sc * 0.35 + 0.5)
    return total


def bits(card_ids):
    mask = 0
    for i in card_ids:
        mask |= BIT_OF[i]
    return mask


def sim_action(hand, deck, sc, target):
    goal = 0 if sc >= target else target
    hand_mask = bits(hand)
    deck_mask = bits(deck)
    pool_mask = hand_mask | deck_mask
    good_enough = (not goal) or sc + optimistic_good(pool_mask) >= goal
    best_play = None
    best_play_v = -10 ** 9

    for score, flags, mask, ids in find_combos(hand):
        after = pool_mask ^ mask
        new_sc = sc + score
        ceil = new_sc + max_points(after)
        if goal and new_sc < goal and ceil < goal:
            continue
        mixed = flags == 4
        if mixed and target == GOLD and good_enough and not (goal and new_sc >= goal) and combo_destroys(ids, pool_mask):
            continue
        v = ceil * 10 + score
        if goal and new_sc >= goal:
            v += 1000000
        if mixed and good_enough and not (goal and new_sc >= goal):
            v -= 250
        if v > best_play_v:
            best_play_v = v
            best_play = (score, flags, mask, ids)

    if best_play and best_play[1] in (1, 2):
        return 1, best_play
    if best_play and best_play_v >= 1000000:
        return 1, best_play
    if len(hand) < 5 and deck:
        return 0, None
    if not hand:
        return -1, None

    best_discard = None
    best_discard_v = -10 ** 9
    if len(hand) == 5 or not deck:
        for card in hand:
            after = pool_mask ^ BIT_OF[card]
            ceil = sc + max_points(after)
            if goal and sc < goal and ceil < goal:
                continue
            rest = [c for c in hand if c != card]
            v = optimistic_good(after) * 4 + seeded_combo_value(rest, after) * 2 + pair_value(color_run_needs(rest, deck_mask))
            if v > best_discard_v:
                best_discard_v = v
                best_discard = card
        if best_discard is None:
            worst = hand[0]
            worst_ceil = 10 ** 9
            for card in hand:
                ceil = sc + max_points(pool_mask ^ BIT_OF[card])
                if ceil < worst_ceil:
                    worst_ceil = ceil
                    worst = card
            best_discard = worst
            best_discard_v = -1

    if best_play and (best_discard is None or best_play_v >= best_discard_v):
        return 1, best_play
    if best_discard is not None:
        return 2, best_discard
    return 0, None


def simulate_rest(hand, used_mask, start_score, base, index, target):
    blocked = used_mask
    for card in hand:
        blocked |= BIT_OF[card]
    deck = ordered_deck([i for i in range(24) if not (blocked & BIT_OF[i])], base, index)
    cur = list(hand)
    sc = start_score
    for _ in range(48):
        while len(cur) < 5 and deck:
            cur.append(deck.pop())
        if not cur:
            break
        kind, payload = sim_action(cur, deck, sc, target)
        if kind <= 0:
            if not deck:
                left = find_combos(cur)
                played = False
                goal = 0 if sc >= target else target
                cur_mask = bits(cur)
                for score, flags, mask, ids in left:
                    after = cur_mask ^ mask
                    ceil = sc + score + max_points(after)
                    if goal and sc < goal and sc + score < goal and ceil < goal:
                        continue
                    if flags in (1, 2) or not deck or (goal and sc + optimistic_good(cur_mask) < goal):
                        sc += score
                        drop = set(ids)
                        cur = [c for c in cur if c not in drop]
                        played = True
                        break
                if not played:
                    break
                continue
            continue
        if kind == 1:
            sc += payload[0]
            drop = set(payload[3])
            cur = [c for c in cur if c not in drop]
        else:
            drop = payload
            cur = [c for c in cur if c != drop]
    return sc


def chance_from(hand, used_mask, sc, target, seeds):
    goal = 0 if sc >= target else target
    silver_hits = gold_hits = target_hits = total = 0
    for base, index in seeds:
        final = simulate_rest(hand, used_mask, sc, base, index, target)
        total += final
        if final >= SILVER:
            silver_hits += 1
        if final >= GOLD:
            gold_hits += 1
        if goal and final >= goal:
            target_hits += 1
    n = max(len(seeds), 1)
    return (
        1.0 if not goal else target_hits / n,
        silver_hits / n,
        gold_hits / n,
        total / n,
    )


def canonical_form(keys):
    cards = []
    for key in keys:
        color, rank = key.rsplit("-", 1)
        cards.append((color, int(rank)))
    best_sig = None
    for perm in COLOR_PERMS:
        to_canon = {COLORS[i]: perm[i] for i in range(3)}
        sig = ",".join(sorted(f"{to_canon[color]}-{rank}" for color, rank in cards))
        if best_sig is None or sig < best_sig:
            best_sig = sig
    return best_sig


def unique_canonical_hands():
    found = []
    seen = set()
    for idxs in combinations(range(24), 5):
        sig = canonical_form([KEY_OF[i] for i in idxs])
        if sig not in seen:
            seen.add(sig)
            found.append(sig)
    return found


def discard_rank(card, cards):
    rest = [c for c in cards if c != card]
    after = ((1 << 24) - 1) ^ BIT_OF[card]
    deck_mask = after ^ bits(rest)
    ply = 0
    draws = [i for i in range(24) if i not in cards]
    for draw in draws:
        locked = [c for c in find_combos(rest + [draw]) if c[1] in (1, 2)]
        ply += max(c[0] for c in locked) if locked else 0
    return (
        optimistic_good(after) * 4
        + seeded_combo_value(rest, after) * 2
        + pair_value(color_run_needs(rest, deck_mask))
        + ply / max(len(draws), 1)
    )


def analyze_hand(sig, target, samples=0):
    cards = [KEY_INDEX[key] for key in sig.split(",")]
    locked = [c for c in find_combos(cards) if c[1] in (1, 2)]
    locked.sort(key=lambda c: (-c[0], 0))
    unknown = [-1, -1, -1, -1, -1, -1, -1, -1]
    if locked:
        play_idx = [sig.split(",").index(KEY_OF[i]) for i in sorted(locked[0][3])]
        return [1, play_idx[0], play_idx[1], play_idx[2], *unknown]
    ranked = sorted(cards, key=lambda card: (-discard_rank(card, cards), KEY_OF[card]))
    return [0, cards.index(ranked[0]), 0, 0, *unknown]


def _worker(args):
    sig, samples = args
    return sig, {
        "300": analyze_hand(sig, SILVER, samples),
        "400": analyze_hand(sig, GOLD, samples),
    }


def write_book(path, data, samples, elapsed, hands):
    payload = {
        "v": 1,
        "samples": 0,
        "mode": "opening-heuristic",
        "hands": hands,
        "ms": int(elapsed * 1000),
        "data": data,
    }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.OKEY_OPENING_BOOK=")
        fh.write(json.dumps(payload, separators=(",", ":")))
        fh.write(";\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--out", default="opening-book.js")
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    hands = unique_canonical_hands()
    if args.limit:
        hands = hands[: args.limit]
    print(f"canonical opening hands: {len(hands)} (from 42504)", flush=True)

    if args.bench:
        sample = hands[: args.bench]
        t0 = time.time()
        for sig in sample:
            analyze_hand(sig, SILVER, args.samples)
        elapsed = time.time() - t0
        per = elapsed / max(len(sample), 1)
        full = len(unique_canonical_hands())
        print(f"bench {len(sample)} hands x {args.samples} samples: {elapsed:.2f}s ({per:.3f}s/hand)")
        print(f"cache sizes max={len(MAX_CACHE)} opt={len(OPT_CACHE)}")
        print(f"estimate full {full} x 2 targets: {per * full * 2 / max(args.jobs, 1):.0f}s with {args.jobs} jobs")
        return

    t0 = time.time()
    data = {"300": {}, "400": {}}
    for i, sig in enumerate(hands, 1):
        data["300"][sig] = analyze_hand(sig, SILVER)
        data["400"][sig] = analyze_hand(sig, GOLD)
        if i % 1000 == 0 or i == len(hands):
            print(f"{i}/{len(hands)} {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    write_book(args.out, data, args.samples, elapsed, len(hands))
    print(f"wrote {args.out} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
