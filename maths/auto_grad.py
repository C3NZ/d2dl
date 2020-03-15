from mxnet import autograd, np, npx

npx.set_np()

# Create the initial ndarrary
x = np.arange(4)

# Allocate memory for the gradient buffer that is of the same shape
# as the input vector.
x.attach_grad()

# Display the gradient buffer, which is initialized at 0. It is initialized
# at 0 for the event that the gradient is accidentally applied to the
# differentiated function before the gradients are computed.
print(x.grad)

# Build the graph and record computations across it
with autograd.record():
    y = 2 * np.dot(x, x)

# Print the reuslt of y
print(y)

# Compute the gradient of y with respect to each component of x
y.backward()

# Teh gradient of y = 2 * dot(x, x) is 4x, which means that
# the gradients of x should be 0, 4, 8, and 12 respectively
print(x.grad)


def detach_computation():
    """
    Lets say we have a function y that relies on a function x. We could write 
    that as: y = x * x

    now lets also say that we have a function z, that depends on both y and x.
    If we wanted to compute the gradient of z with respect to x and treat y as a 
    constant, how could we possibly do it?

    The answer, we can compute y and then detach it from the graph, allowing
    it to be used as a constant in another computation

    """
    x = np.arange(4)
    x.attach_grad()

    with autograd.record():
        # The computation for y is computed, with the derivative of y
        # being 2x.
        y = x * x
        # Here we convert y to a constant by creating a variable u that
        # is detached from the graph, allowing us to record future computations
        # that store the result of y but not how the results came about.
        u = y.detach()
        # Now we can compute the gradient of z with respect to x while treating
        # u as a constant. (The derivative becomes u*x instead of 3x^2, which
        # would've been computed had we used y instead of u).
        z = u * x

    print("z")

    # We are now obtaining the gradients of function z with respect to x. Because
    # we used the detached variable u, the gradient becomes the result of u * x.
    z.backward()
    print(x.grad)
    print("The gradients for the partial derivate of z with respect to x", x.grad)
    print(x.grad == u)

    # Because the operations for y were recorded when computing underneath
    # the autograd, we're able to compute the gradient for y with respect
    # to x.
    y.backward()
    print("The derivative of y is 2x, resulting in gradients: ", x.grad)
    print(x.grad == 2 * x)


def compute_gradient_with_control_flow():
    """
    How can we handle computing the gradient of a functino that introduces
    control flow to our program?
    """

    def f(a):
        """
        A piecewise mathematical function that is used for computing some
        arbritrary values.
        """
        b = 2 * a

        # Multiply each element by 2 until
        # the l2 norm of b is greater than 1000
        while np.linalg.norm(b) < 1000:
            b = 2 * b

        # If the sum of b is greater than 0, c = b, else
        # c = 100 * b.
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b

        return c

    # Create a random normal distribution and initialize it's gradient array.
    a = np.random.normal()
    a.attach_grad()

    # Record all gradients when computing the result of f(a)
    with autograd.record():
        d = f(a)

    # Backward propagte d in order to obtain the gradients for variables a.
    d.backward()

    # In f(a), we're simply multiplying a by some scalar number that is
    # determined by the initial random normal distribution. We can
    # mathematically formula f(a) = k * a, where k is the scalar multiplied
    # by a. To verify our gradient, we can simply check if f(a) / k = a.
    print("f(a) / k = a: ", a.grad == d / a)


if __name__ == "__main__":
    detach_computation()
    compute_gradient_with_control_flow()
