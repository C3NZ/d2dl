import random

from mxnet import np, npx

import d2l

npx.set_np()


def sample_from_multinomial():
    # Take one sample from the fair probability distribution. Right now, this
    # represents a dice
    fair_probs = [1.0 / 6] * 6
    choice = np.random.multinomial(1, fair_probs)
    print(choice)

    # Choose ten random samples with 3 different experiments and then return a
    # matrix of the three experiment vectors showing the results.
    choices = np.random.multinomial(10, fair_probs, size=3)
    print(choices)

    # Lets sample 1000 times and determine the probability of each sample
    # occurring. We save the result as a 32 bit float for computation.
    counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
    print(counts / 1000)


def visualize_probability_convergence():
    """
    Visualize the convergence of sampling 
    """
    fair_probs = [1.0 / 6] * 6

    # Run 500 experiments where you roll the dice ten times.
    counts = np.random.multinomial(10, fair_probs, size=500)

    # Summarize all of the counts in each column for each row.
    cum_counts = counts.astype(np.float32).cumsum(axis=0)

    # Compute the estimate probability for each dice being rolled through
    # all of the experiments. This will converge towards 16%
    estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

    # Plot each estimated probability
    d2l.set_figsize((6, 4.5))
    for i in range(6):
        d2l.plt.plot(estimates[:, i].asnumpy(), label=f"P(die={str(i+1)})")

    #  Add the true probability of you rolling any dice number
    d2l.plt.axhline(y=0.167, color="black", linestyle="dashed")

    # Set the x and y label for the current axes
    d2l.plt.gca().set_xlabel("Groups of experiments")
    d2l.plt.gca().set_ylabel("Estimated probability")

    # Create the legend and save the figure as an image.
    d2l.plt.legend()
    d2l.plt.savefig("probability_convergence.png")


if __name__ == "__main__":
    sample_from_multinomial()
    visualize_probability_convergence()
