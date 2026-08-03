# Result integrity rules

1. Never label a committed JSON file as independently reproduced unless a clean runner generated it from the same commit.
2. Store command, seed, architecture, dependency versions, wall time, and metrics in every result.
3. Keep negative and failed results when they falsify a selected method.
4. Do not infer a private optimizer or dataset from final weights alone.
5. Separate architecture compatibility, low-bit invariant correctness, and behavior retention.
6. Require real-loss validation for discrete changes proposed by a local approximation.
7. Report seed-level values, not only aggregate means.
8. Do not convert a confidence estimate into a fact statement.
