from IPython import display
from mxnet import np, npx

import d2l

npx.set_np()


def numerical_func(x):
    """
    A random numerical function
    """
    return (3 * x ** 2) - (4 * x)


def calculate_numerical_limit(func, x, h):
    """
        Calculate the numerical limit of a function given the function, the 
        x value, and the value we're approaching.
    """
    return (func(x + h) - func(x)) / h


def find_limit():
    """
        Find the limit of x=1 as h approaches 0.
    """
    h = 0.1
    for _ in range(5):
        numerical_limit = calculate_numerical_limit(numerical_func, 1, h)
        print(f"h = {h:.5f}, numerical limit = {numerical_limit:.5f}")
        h *= 0.01


def use_svg_display():
    """
    Setup matplotlib to use svg format
    """
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the figure size
    """
    use_svg_display()
    d2l.plt.rcParams["figure.figsize"] = figsize


def configure_axes(axes, x_label, y_label, x_limit, y_limit, x_scale, y_scale, legend):
    """
    Configure the matplot lib axes
    """
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_xscale(x_scale)
    axes.set_yscale(y_scale)
    axes.set_xlim(x_limit)
    axes.set_ylim(y_limit)

    if legend:
        axes.legend(legend)

    axes.grid()


def plot(
    X,
    Y=None,
    x_label=None,
    y_label=None,
    x_limit=None,
    y_limit=None,
    x_scale="linear",
    y_scale="linear",
    fig_size=(3.5, 2.5),
    fmts=["-", "m--", "g-", "r:"],
    legend=[],
    axes=None,
):
    d2l.set_figsize(fig_size)
    axes = axes if axes else d2l.plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1) or (
            isinstance(X, list) and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]

    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]

    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()

    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)

    configure_axes(axes, x_label, y_label, x_limit, y_limit, x_scale, y_scale, legend)


x = np.arange(0, 3, 0.1)
plot(
    x,
    [numerical_func(x), 2 * x - 3],
    "x",
    "f(x)",
    legend=["f(x)", "Tangent line (x=1)"],
)

if __name__ == "__main__":
    find_limit()
