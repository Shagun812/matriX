# Valency MCP usage for matriX

This project uses Valency MCP as a builder-side literature and provenance tool. It is not part of the runtime diagnosis loop, and it is not used to alter the benchmark information set during evaluation.

## Purpose

Valency MCP is used to:

- source and validate TiO2 reference constants
- check anatase/rutile XRD assumptions against external literature
- review optical-gap conventions and fitting assumptions
- verify prior-art claims and benchmark positioning
- maintain a transparent provenance trail for scientific assumptions used in simulator design and documentation

## Important boundary

The model should not use Valency during the actual diagnosis pipeline. The benchmark must remain deterministic and reproducible, with evaluation based on fixed evidence, fixed checks, and fixed simulator logic.

The correct usage is:

- builder / researcher: uses Valency MCP to validate and source scientific assumptions
- runtime workflow: uses only the repo-defined evidence and logic

## Recommended wording

> Valency MCP is used as a builder-side literature and provenance tool to source and verify TiO2 constants, prior-art references, and simulator assumptions. It is not part of the benchmark’s runtime information set for diagnosis. The benchmark uses only agent-visible measurements, checks, and fixed simulator logic, while MCP supports human validation of scientific grounding outside the decision loop.

## Copilot CLI setup

Use the Valency server in the local Copilot CLI environment:

```bash
copilot mcp list
copilot mcp get valency
```

If it is not configured:

```bash
copilot mcp add --transport http valency https://mcp.valency.io/
```

## Good use cases

- checking TiO2 reference patterns
- reviewing simulator assumptions before freezing them
- validating prior-art and novelty statements
- supporting a documented provenance trail for constants and scientific claims

## Not recommended

- live literature lookup inside the diagnosis loop
- altering check selection or priors based on online retrieval during evaluation
- storing API credentials or auth tokens in the repository
- treating Valency as a runtime inference subsystem for the benchmark
