# dt

Run with: `python3 dt.py` in the same directory as 'car.data'.

dt contains an implementation of an ID3 decicion-tree that partitions based on
information gain.

The decision-tree here classifies the 'acceptability' of a car into
unacceptable, acceptable, good, and very good. The data is held in 'car.data'
with attributes buying price, maintenance cost, door count, people held, size of
trunk, and safety.

The program runs twice. During each iteration, it partitions the data into 75%
training data and 25% test data. It then trains the tree and outputs how long
that takes. Then it evaluates the tree and outputs how long that takes. Then it
outputs the evaluation of the tree.
