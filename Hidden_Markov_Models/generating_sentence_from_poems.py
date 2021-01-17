import numpy as np
import string
import sys

# for first word of the phrase
initial = {} 
# will store (first, second) word of the phrase
second_word = {}
# will store (token[t - 2], token[t - 1]) => token[t]
# for the last two words the key will be token END, (token[T - 2], token(T - 1)) => END 
transitions = {}

# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
	remove_punctuation = remove_punctuation_2
else:
	remove_punctuation = remove_punctuation_3

# d : dictionary to update
# k : key 
# v: value
def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

for line in open('poems_of_robert_frost.txt'):
	tokens = remove_punctuation(line.rstrip().lower()).split()

	T = len(tokens)
	for i in range(T):
		t = tokens[i]
		if i == 0:
		# measure the distribution of the first word
			initial[t] = initial.get(t, 0.) + 1
		else:
			t_1 = tokens[i-1]
			#  for the last two words
			if i == T - 1:
			    # measure probability of ending the line
			    add2dict(transitions, (t_1, t), 'END')
			# for the second words
			if i == 1:
			    # measure distribution of second word
			    # given only first word
			    add2dict(second_word, t_1, t)
			else:
				# for the remaining cases
			    t_2 = tokens[i-2]
			    add2dict(transitions, (t_2, t_1), t)

# for d, v in initial.items():
# 	print(d, v)

# for d, v in second_word.items():
# 	print(d, v)

# for d, v in transitions.items():
# 	print(d, v)

# normalize the distributions
initial_total = sum(initial.values())
for t, c in initial.items():
    initial[t] = c / initial_total

def list2pdict(ts):
    # turn each list of possibilities into a dictionary of probabilities
	d = {}
	n = len(ts)
	for t in ts:
	    d[t] = d.get(t, 0.) + 1
	for t, c in d.items():
	    d[t] = c / n
	return d

for t_1, ts in second_word.items():
	# replace list with dictionary of probabilities
	second_word[t_1] = list2pdict(ts)

for k, ts in transitions.items():
	transitions[k] = list2pdict(ts)


# for d, v in initial.items():
# 	print(d, v)

# for d, v in second_word.items():
# 	print(d, v)

# for d, v in transitions.items():
# 	print(d, v)

# generate 4 lines
def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t

def generate():
    for i in range(4):
        sentence =[]

        # initial word
        w0 = sample_word(initial)
        sentence.append(w0)

        # sample second word
        w1 = sample_word(second_word[w0])
        sentence.append(w1)

        # second-order transitions until END
        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sentence))

generate()




















