from src.platform.http.errors import DomainError


class FormulaError(DomainError):
    pass


def not_found() -> FormulaError:
    return FormulaError("formula_not_found", "Formula not found", 404)
