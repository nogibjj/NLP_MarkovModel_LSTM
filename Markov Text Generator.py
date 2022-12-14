# Bare Bones Markov Text Generator

# import modules
import random
import nltk

# define function
def finish_sentence(sentence, n, corpus, deterministic=False):
    # Create a separate function to build the count dictionary
    count = buildCount(sentence, n, corpus)
    # If the count dictionary is empty, create a new variable for the back-off
    if count == {}:
        x = n
    # Create a back-off loop for while the count dictionary is empty
    while (count == {}) and (x > 0):
        count = buildCount(sentence, x - 1, corpus)
        x -= 1
        pass
    # Create a deterministic append to the sentence
    if deterministic:
        sentence.append(max(count, key=count.get))
        pass
    # Create a random append to the sentence
    else:
        sentence.append(
            random.choices(
                list(count.keys()),
                # use a list comprehension to define the weights
                weights=[
                    eachCount / sum(count.values()) for eachCount in count.values()
                ],
                k=1,
            )[0]
        )
        pass
    pass


# define loop function to allow for back-off recursion
def buildCount(sentence, n, corpus):
    count = {}
    # Count the frequency of each word if n is 1
    if n == 1:
        for eachWord in corpus:
            if eachWord not in count:
                count[eachWord] = 1
                pass
            else:
                count[eachWord] += 1
                pass
            pass
        pass
    else:
        # Loop over the entire corpus
        for eachToken in range(n - 1, len(corpus)):
            # find the n-grams that match the sentence
            if corpus[eachToken - n + 1 : eachToken] == sentence[-n + 1 :]:
                # Count the number of times each n-gram occurs
                if corpus[eachToken] not in count:
                    count[corpus[eachToken]] = 1
                    pass
                else:
                    count[corpus[eachToken]] += 1
                    pass
                pass
            pass
        pass
    return count


# define main function
if __name__ == "__main__":
    # Test the function
    sentence = ["how", "many", "days", "in", "a"]
    n = 5
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    deterministic = False
    while (sentence[-1] not in [".", "!", "?"]) and (len(sentence) < 10):
        finish_sentence(sentence, n, corpus, deterministic)
        pass
    print(sentence)
