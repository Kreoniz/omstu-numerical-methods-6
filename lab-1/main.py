from decimal import Decimal, getcontext
from rich.console import Console
from rich.table import Table

getcontext().prec = 25
console = Console()

X = Decimal("15.0897")
a = Decimal("11.7")
b = Decimal("0.0937")
c = Decimal("5.081")

table = Table(title="Исходные данные", show_lines=True)

table.add_column("Параметр", style="cyan")
table.add_column("Значение", style="magenta")

table.add_row("X", str(X))
table.add_row("a", str(a))
table.add_row("b", str(b))
table.add_row("c", str(c))
table.add_row("Формула Z", "(a - c)^2 / (√a + 3b)")

console.print(table)

diff = a - c
numerator = diff**2
sqrt_a = a.sqrt()
denominator = sqrt_a + 3 * b
Z = numerator / denominator

console.print("\n[bold]Ход вычислений:[/bold]")
console.print(f"a - c = {diff}")
console.print(f"(a - c)^2 = {numerator}")
console.print(f"√a = {sqrt_a}")
console.print(f"√a + 3b = {denominator}")
console.print(f"\n[bold green]Z = {Z}[/bold green]")
