"""Generate a synthetic, realistic-style telecom service-ticket dataset.

Why synthetic?
    Real operator tickets contain customer PII and are never published. This
    generator reproduces the *shape* of the routing problem seen at a
    provincial telecom operator's ITSM desk: free-text notes written by
    front-line customer-service agents, which must be routed to one of several
    resolver groups.

What makes it non-trivial (so a model cannot hit 100%):
    * Shared vocabulary across departments ("broadband", "order", "fee" appear
      in several classes).
    * Cross-department side remarks (a billing complaint that also
      mentions the broadband being slow).
    * Agent shorthand, typos, inconsistent casing, missing punctuation.
    * Generic filler ("customer very upset, please handle asap").
    * A small amount of label noise, mimicking tickets that were routed to the
      wrong group in the historical log.

Usage:
    python data/generate_tickets.py --n 6000 --seed 42 --out data/tickets.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

DEPARTMENTS = [
    "BILLING",          # charges, invoices, package fees
    "CRM",              # customer profile, orders, account changes in CRM
    "OSS_PROVISIONING", # service activation / work orders in OSS
    "NETWORK_OPS",      # signal, outage, speed - network side
    "PAYMENT",          # top-up, refunds, payment channel issues
    "FIELD_SERVICE",    # installation & on-site repair visits
]

# Class prior: billing and network complaints are the most common in practice.
CLASS_WEIGHTS = [0.22, 0.18, 0.16, 0.20, 0.12, 0.12]

SLOTS = {
    "plan": ["unlimited 5G plan", "family bundle", "199 plan", "basic 4G plan",
             "fiber+mobile combo", "data add-on", "intl roaming pack", "prepaid plan"],
    "amount": ["30", "58", "99", "120", "15.5", "200", "8", "45"],
    "month": ["last month", "this month", "September", "October", "the past 2 months"],
    "service": ["broadband", "fiber", "5G", "IPTV", "landline", "mobile data", "VoLTE"],
    "channel_app": ["the app", "WeChat", "Alipay", "online banking", "the website", "a store counter"],
    "area": ["Yanta district", "Beilin", "Weiyang", "Chang'an", "Lianhu", "the high-tech zone",
             "Baqiao", "a rural village"],
    "days": ["2", "3", "5", "a week", "10", "several"],
    "doc": ["ID", "passport", "business license"],
}

TEMPLATES = {
    "BILLING": [
        "customer charged {amount} extra on {month} bill for {plan}, says never subscribed",
        "bill shows duplicate monthly fee for {plan}, requests correction",
        "invoice amount does not match package price, {plan} should be cheaper",
        "cust disputes data overage charge of {amount}, claims usage was low",
        "monthly fee still deducted after {plan} cancelled {month}",
        "unknown value-added service fee {amount} on bill pls check",
        "e-invoice cannot be issued for {month}, amount looks wrong",
        "discount for {plan} not applied, billed at full price",
        "roaming charges {amount} appeared though cust did not travel",
        "package changed but still billed under old {plan} rate",
        "bill cycle wrong, charged twice in {month}",
        "rated usage incorrect, {service} billed at wrong tariff",
    ],
    "CRM": [
        "customer name on account incorrect, update with {doc} failed in system",
        "cannot change account owner, CRM page shows error on submit",
        "order for {plan} stuck in 'pending review' status for {days} days",
        "contact number change not saved, customer profile still old",
        "real-name verification failed although {doc} is valid",
        "duplicate customer records found, cannot merge accounts",
        "agent cannot open customer 360 view, system timeout",
        "order cancelled by system automatically, cust did not request",
        "membership points missing after {plan} upgrade",
        "customer level shows wrong, VIP flag lost",
        "service request created but order number not generated",
        "address change submitted but CRM still shows old address",
    ],
    "OSS_PROVISIONING": [
        "{service} paid and order completed but service not activated",
        "work order for {service} activation stuck at provisioning step",
        "number port-in failed, provisioning returned error code",
        "new SIM activated in CRM but HLR not updated, no service",
        "{plan} changed but network side still old speed profile",
        "IPTV account not opened after order, provisioning pending",
        "suspension lifted but line still barred, resync needed",
        "data add-on purchased, quota not provisioned",
        "resource allocation failed, no port available for {service}",
        "work order returned by system, status mismatch between crm and oss",
        "VoLTE function not enabled after {plan} activation",
        "service barred by mistake after payment, need reactivation",
    ],
    "NETWORK_OPS": [
        "no signal at home in {area} since {days} days",
        "{service} very slow at night in {area}, speed test far below plan",
        "frequent call drops in {area}, multiple users nearby same issue",
        "{service} down in whole building in {area}",
        "5G icon shows but cannot access internet in {area}",
        "high latency and packet loss on {service}, gaming impossible",
        "base station near {area} seems down, many complaints",
        "cannot make calls indoors in {area}, weak coverage",
        "{service} disconnects every few minutes, modem lights normal",
        "mobile data unstable in {area} during peak hours",
        "dns resolution failure, some websites not opening on {service}",
        "large area outage reported in {area}",
    ],
    "PAYMENT": [
        "topped up {amount} via {channel_app} but balance not updated",
        "payment deducted twice through {channel_app}, requests refund",
        "refund of {amount} promised {month} still not received",
        "auto-pay failed though bank account has balance",
        "recharge card number shows already used",
        "paid bill via {channel_app}, account still in arrears and suspended",
        "cust wants refund for overpayment of {amount}",
        "payment success page but no transaction record in system",
        "deposit not returned after contract ended",
        "cannot pay through {channel_app}, error at checkout",
        "balance transfer between numbers failed",
        "top-up promotion bonus {amount} not credited",
    ],
    "FIELD_SERVICE": [
        "installation appointment for {service} missed, technician did not show",
        "need on-site visit, fiber cable cut near house in {area}",
        "requested modem replacement, device broken",
        "technician visit rescheduled {days} times, customer angry",
        "new {service} install pending, no engineer assigned",
        "optical modem red LOS light, needs site check in {area}",
        "move {service} to new address, need relocation install",
        "set-top box faulty, request engineer to replace",
        "wall socket for fiber damaged during renovation",
        "outdoor drop cable hanging low, safety concern in {area}",
        "engineer left without finishing installation",
        "router installation incomplete, wifi not configured",
    ],
}

# Side remarks written in the vocabulary of a given department. A ticket of
# department D gets a remark from some *other* department E with some
# probability, so the text contains misleading keywords (a billing complaint
# that also mentions slow broadband) - as real tickets do. Because remarks are
# drawn from other departments, they never become a shortcut for the true label.
CROSS_MENTIONS = {
    "BILLING": ["customer also mentioned bill is high", "asks for compensation on bill",
                "wants a fee waiver"],
    "CRM": ["customer threatens to cancel {plan}", "wants to upgrade to {plan} later",
            "update contact number too"],
    "OSS_PROVISIONING": ["order shows completed in CRM", "service status looks inactive",
                         "line may be barred"],
    "NETWORK_OPS": ["also {service} a bit slow lately", "no signal on {service} since then",
                    "neighbours have same problem"],
    "PAYMENT": ["cust paid via {channel_app} on time", "service suspended due to arrears",
                "customer already paid the bill"],
    "FIELD_SERVICE": ["modem was replaced last year", "engineer visited before",
                      "cable looks old"],
}

FILLERS = [
    "", "", "", "",
    "customer very upset.", "pls handle asap.", "repeat complaint.",
    "VIP customer.", "cust called 3 times.", "urgent!!", "escalated by supervisor.",
    "cust requests callback.", "2nd contact this week.", "threatens complaint to regulator.",
]

TYPO_MAP = {"customer": ["cust", "custmer", "customer"], "please": ["pls", "plz", "please"],
            "service": ["svc", "service", "serivce"], "account": ["acct", "account"],
            "not": ["not", "nt"], "received": ["recieved", "received"]}


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, values in SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(values), 1)
    return out


def add_noise(text: str, rng: random.Random) -> str:
    words = text.split()
    for i, w in enumerate(words):
        base = w.lower().strip(".,!")
        if base in TYPO_MAP and rng.random() < 0.25:
            words[i] = rng.choice(TYPO_MAP[base])
    text = " ".join(words)
    r = rng.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.45:
        text = text[0].upper() + text[1:]
    if rng.random() < 0.3:
        text = text.rstrip(".") + "."
    return text


def make_ticket(dept: str, rng: random.Random) -> tuple[str, int]:
    template_id = rng.randrange(len(TEMPLATES[dept]))
    parts = [fill(TEMPLATES[dept][template_id], rng)]
    extras = []
    if rng.random() < 0.30:
        other = rng.choice([d for d in DEPARTMENTS if d != dept])
        extras.append(fill(rng.choice(CROSS_MENTIONS[other]), rng))
    filler = rng.choice(FILLERS)
    if filler:
        extras.append(filler)
    rng.shuffle(extras)
    if extras and rng.random() < 0.2:  # sometimes the agent writes context first
        parts = extras[:1] + parts + extras[1:]
    else:
        parts = parts + extras
    sep = rng.choice([", ", ". ", "; ", " - "])
    if sep != ", ":
        parts = [p.rstrip(".") for p in parts]
    return add_noise(sep.join(parts), rng), template_id


def generate(n: int, seed: int, label_noise: float) -> list[dict]:
    rng = random.Random(seed)
    start = datetime(2023, 1, 1)
    rows = []
    for i in range(n):
        dept = rng.choices(DEPARTMENTS, weights=CLASS_WEIGHTS)[0]
        text, template_id = make_ticket(dept, rng)
        label = dept
        if rng.random() < label_noise:  # historically mis-routed ticket
            label = rng.choice([d for d in DEPARTMENTS if d != dept])
        created = start + timedelta(minutes=rng.randint(0, 365 * 24 * 60))
        rows.append({
            "ticket_id": f"TK{100000 + i}",
            "created_at": created.strftime("%Y-%m-%d %H:%M"),
            "channel": rng.choices(["hotline", "app", "store", "wechat"], [0.5, 0.25, 0.15, 0.1])[0],
            "description": text,
            "department": label,
            # Evaluation-only column: which phrasing template produced the text.
            # Used for the "unseen phrasing" split; never used as a feature.
            "template_id": f"{dept}_{template_id:02d}",
        })
    rows.sort(key=lambda r: r["created_at"])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-noise", type=float, default=0.03)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "tickets.csv")
    a = p.parse_args()
    rows = generate(a.n, a.seed, a.label_noise)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} tickets -> {a.out}")


if __name__ == "__main__":
    main()
