# Chapter 1

1. Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself. What do you think would happen in this case? Would it learn a different way of playing?
    - If a reinforcement learning algorithm only played against random opponents it would probably only learn how to play specifically against those opponents (and not generally). Playing against itself would bypass this restrictions and probably allow it to play more optimally/at a higher level (we see some of this in some of OpenAI's achievements).

2. *Symmetries* Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the reinforce- ment learning algorithm described above to take advantage of this? In what ways would this improve it? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?
    - I think that