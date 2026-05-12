## Development

Use `uv` to manage package dependencies, tests, and run scripts.

```
uv run python <script>
uv run python -m pytest
uv add <package-name>
uv add --dev <package-name>
```

## Hygiene

1. Error early and provide informative error messages, don't be clever
2. Avoid deep inheritance hierarchies
3. Keep boilerplate minimal
4. Avoid defining the same default value in multiple places
5. If a string literal is used more than once, it should be a named constant
6. Use modern type annotation: `d: dict[str, int]`, `v: str | None`
