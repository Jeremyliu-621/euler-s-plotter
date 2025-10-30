"""
euler_plot.py — Visualize Euler’s Method (beginner-friendly)

How to run:
    Save this file as euler_plot.py
    In a terminal or command prompt, run:
        python euler_plot.py

What this program does:
    • Asks you for:
        - A derivative function f(y, t) as a Python expression (e.g., -2*y + t)
        - An exact solution g(t) as a Python expression (e.g., t - 0.5 + 0.5*exp(-2*t))
        - A time step delta t (e.g., 0.1)
        - A start time t0 (press Enter to use 0)
        - An end time t_end (e.g., 5)
        - A starting value y0 at t0 (e.g., 1)
    • Computes Euler’s Method from t0 to t_end with your delta t
    • Plots:
        - The Euler approximation as connected points/line
        - The exact function g(t) as a smooth curve (if it can be evaluated)
    • Prints the first few (t, y_euler) values in a small table
    • (Nice-to-have) Prints the maximum absolute error at the Euler grid points
      by comparing Euler y’s to g(t) at the same t’s (if g(t) can be evaluated)

Safety notes for expressions:
    We evaluate your math expressions using a very small “allowed” dictionary
    (sin, cos, exp, sqrt, pi, etc.) and block Python built-ins. If you make a
    mistake in your expression, we’ll show a friendly error and let you try again.

Example inputs you can paste:
    f(y, t) = -2*y + t
    g(t)    = t - 0.5 + 0.5*exp(-2*t)
    delta t = 0.1
    t0      = 0
    t_end   = 5
    y0      = 1
"""

import math
import sys
import matplotlib.pyplot as plt


def prompt_expression(name, var_hint):
    """
    Ask the user for a Python expression and compile it safely.

    name: a label like "f(y, t)" or "g(t)" for prompts and error messages
    var_hint: a string like "use variables y and t" shown to the user

    Returns a tuple: (compiled_code_object, original_text)
    """
    while True:
        print("")
        print("Enter", name)
        print("  Tip:", var_hint)
        expr = input("  >>> ").strip()

        if expr == "":
            print("  You did not enter anything. Please try again.")
            continue

        # Try to compile the expression first (this catches simple syntax errors)
        try:
            code = compile(expr, "<user {}>".format(name), "eval")
            return code, expr
        except Exception as e:
            print("  There was a problem with your expression:")
            print("   ", repr(e))
            print("  Please try again.")


def make_safe_eval_env():
    """
    Build the restricted environment dictionary allowed during eval().
    We do NOT expose Python builtins. Only the names below are allowed.

    You can extend this with other math functions if you want.
    """
    env = {
        # Math constants
        "pi": math.pi,
        "e": math.e,

        # Common math functions
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,     # natural log
        "log10": math.log10,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "floor": math.floor,
        "ceil": math.ceil,
        "fabs": math.fabs,
        "pow": math.pow,
        "abs": abs,          # abs is safe here
        "min": min,          # basic helpers
        "max": max,
    }
    return env


def build_f_function(f_code):
    """
    Wrap the compiled code object for f(y, t) into a real Python function:
        def f(y, t): return <your expression>

    We evaluate with a restricted environment that only contains allowed math.
    """
    allowed = make_safe_eval_env()

    def f(y, t):
        # Locals provide the current values of y and t to the expression
        local_vars = {"y": y, "t": t}
        # Globals provide allowed math names; builtins are disabled
        return eval(f_code, {"__builtins__": None, **allowed}, local_vars)

    return f


def build_g_function(g_code):
    """
    Wrap the compiled code object for g(t) into a real Python function:
        def g(t): return <your expression>

    We evaluate with the same restricted environment.
    """
    allowed = make_safe_eval_env()

    def g(t):
        local_vars = {"t": t}
        return eval(g_code, {"__builtins__": None, **allowed}, local_vars)

    return g


def prompt_float(prompt_text, allow_empty=False, default_value=None):
    """
    Ask the user for a floating-point number.
    If allow_empty is True and the user hits Enter, return default_value.
    """
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and allow_empty:
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number (e.g., 0, 1.5, -2, 3.14).")


def prompt_positive_float(prompt_text):
    """
    Ask for a positive floating-point number (> 0). Repeat until valid.
    """
    while True:
        value = prompt_float(prompt_text)
        if value <= 0.0:
            print("  The value must be positive (greater than 0). Please try again.")
        else:
            return value


def euler_method(f, t0, y0, t_end, delta_t):
    """
    Compute Euler's Method with the exact procedure requested:

    Start with (t, y) = (t0, y0).
    While t < t_end:
        y_next = y + delta_t * f(y, t)
        t_next = t + delta_t
        Append (t_next, y_next) to lists.
        Move to the next step.
    """
    t_values = []
    y_values = []

    # Initialize with the starting point
    t = t0
    y = y0
    t_values.append(t)
    y_values.append(y)

    # Step forward until we reach or pass t_end (we stop at the last t < t_end)
    # If t_end lines up exactly with the step size, the final appended point will be t_end.
    while t < t_end:
        try:
            slope = f(y, t)
        except Exception as e:
            print("  There was an error evaluating f(y, t) at y={}, t={}:".format(y, t))
            print("   ", repr(e))
            print("  Stopping the Euler computation early.")
            break

        y_next = y + delta_t * slope
        t_next = t + delta_t

        t_values.append(t_next)
        y_values.append(y_next)

        t = t_next
        y = y_next

    return t_values, y_values


def smooth_exact_curve(g, t0, t_end, num_points=200):
    """
    Create a smoother set of times for the exact solution curve.
    We avoid numpy to keep things simple. We generate evenly spaced values.

    Returns two lists: T_smooth, G_smooth.
    If g(t) fails anywhere, we raise and handle it outside.
    """
    T = []
    Y = []
    if num_points < 2:
        num_points = 2

    # Even spacing including both t0 and t_end
    step = (t_end - t0) / float(num_points - 1)
    i = 0
    while i < num_points:
        t = t0 + i * step
        try:
            y = g(t)
        except Exception as e:
            # Re-raise to be handled by the caller
            raise
        T.append(t)
        Y.append(y)
        i += 1

    return T, Y


def print_sample_table(t_values, y_values, count=10):
    """
    Print the first few (t, y) pairs in a simple table for the user to see.
    """
    print("")
    print("Sample of the Euler (t, y) values:")
    print("{:>12s}  {:>16s}".format("t", "y_euler"))
    print("-" * 31)

    n = len(t_values)
    if n < count:
        count = n

    i = 0
    while i < count:
        print("{:12.6f}  {:16.8f}".format(t_values[i], y_values[i]))
        i += 1

    if n > count:
        print("... ({} more points)".format(n - count))


def compute_max_error(g, t_values, y_values):
    """
    Compute the maximum absolute error at the Euler grid points:
        max |y_euler(t_i) - g(t_i)|
    Returns (max_error, num_points_compared)

    If g fails to evaluate at any point, we raise the exception and handle it outside.
    """
    max_err = 0.0
    compared = 0

    i = 0
    while i < len(t_values):
        t = t_values[i]
        y_e = y_values[i]
        # Evaluate g(t) and update max error
        y_true = g(t)
        err = abs(y_e - y_true)
        if err > max_err:
            max_err = err
        compared += 1
        i += 1

    return max_err, compared


def main():
    print("=== Euler’s Method vs Exact Solution (Matplotlib) ===")

    # 1) Ask for f(y, t) expression and build the function
    f_code, f_text = prompt_expression(
        name="f(y, t)  (derivative dy/dt)",
        var_hint="use variables y and t, e.g., -2*y + t"
    )
    f = build_f_function(f_code)

    # 2) Ask for g(t) expression and build the function
    g_code, g_text = prompt_expression(
        name="g(t)  (exact/ideal solution in terms of t)",
        var_hint="use variable t, e.g., t - 0.5 + 0.5*exp(-2*t)"
    )
    g = build_g_function(g_code)

    # 3) Ask for delta t (positive)
    delta_t = prompt_positive_float("Enter delta t (time step, must be > 0): ")

    # 4) Ask for t0 (allow empty for default 0.0)
    t0 = prompt_float("Enter start time t0 (press Enter for 0): ", allow_empty=True, default_value=0.0)

    # 5) Ask for t_end and validate that t_end > t0
    while True:
        t_end = prompt_float("Enter end time t_end (must be > t0): ")
        if t_end <= t0:
            print("  t_end must be greater than t0. Please try again.")
        else:
            break

    # 6) Ask for y0
    y0 = prompt_float("Enter starting value y0 at t0: ")

    # 7) Compute Euler’s Method sequence
    print("")
    print("Computing Euler steps...")
    t_vals, y_vals = euler_method(f, t0, y0, t_end, delta_t)
    print("  Done. Computed {} points.".format(len(t_vals)))

    # 8) Print a small table of the first few values
    print_sample_table(t_vals, y_vals, count=10)

    # 9) Prepare the exact solution smooth curve (200 points)
    have_exact_curve = True
    T_exact = []
    Y_exact = []
    try:
        T_exact, Y_exact = smooth_exact_curve(g, t0, t_end, num_points=200)
    except Exception as e:
        have_exact_curve = False
        print("")
        print("Could not generate the smooth exact curve g(t):")
        print(" ", repr(e))
        print("We will still show the Euler approximation.")

    # 10) Plot both curves
    plt.figure()
    # Euler approximation as connected line with markers
    plt.plot(t_vals, y_vals, marker="o", linestyle="-", label="Euler approximation")

    # Exact curve if available
    if have_exact_curve:
        plt.plot(T_exact, Y_exact, linestyle="-", label="Exact solution g(t)")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Euler’s Method vs Exact Solution")
    plt.grid(True)
    plt.legend()

    # 11) (Nice-to-have) Compute and print maximum absolute error at grid points
    #     We wrap in try/except in case g(t) fails at the Euler grid points.
    try:
        max_err, compared = compute_max_error(g, t_vals, y_vals)
        print("")
        print("Maximum absolute error at Euler grid points ({} points): {:.6g}".format(compared, max_err))
    except Exception as e:
        print("")
        print("Could not compute max error at Euler grid points:")
        print(" ", repr(e))

    # 12) Show the plot last
    plt.show()

    print("")
    print("Thanks for using the Euler plotter!")
    print("f(y, t) you entered: ", f_text)
    print("g(t) you entered:    ", g_text)


if __name__ == "__main__":
    main()
