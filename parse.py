#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple
from copy import deepcopy

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.
        `progress` says whether to display progress bars as we parse.
        PARSING - Kev:
        Add weight table(dictionary?) that keeps the lowest-weight dotted rule.
        The same dotted rule with diff start and/or end indices should be treated as a diff rule
        """
        self.predicted_cats: Dict[(int, str), bool] = {}  # dict of already predicted classes
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.n_tokens = len(self.tokens)
        self.n_cols = self.n_tokens + 1

        self.cols: List[Agenda]
        self._run_earley()  # run Earley's algorithm to construct self.cols

    def accepted(self):
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():  # the last column
            if (item.rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item.next_symbol() is None  # that is complete
                    and item.start_position == 0):  # and started back at position 0
                return True
        return False  # we didn't find any appropriate item

    def find_best_sentence(self):
        """
        Returns the ROOT -> ... rule in the final col of the chart
        with the lowest weight
        """
        # Look at every completed ROOT in the final col, choose the one with lowest weight
        w = {}  # key is weight of completed ROOTs, val is idx of that rule in self.weights
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item.next_symbol() is None  # that is complete
                    and item.start_position == 0):  # and started back at position 0
                w[item.weight] = item  # get its weight

        best_item = w[min(w)]
        best_weight = best_item.weight

        return best_item, best_weight

    def print_item(self, item):
        if isinstance(item, str):
            # if item is a terminal, just return it
            return item + ' '

        # else go into the backpointers
        terminals = ''  # the terminals we return
        for pointer in item.backpointers:
            terminals += self.print_item(pointer)

        # get rid of trailing space after nested expansions
        terminals = terminals[:-1] if terminals[-2:] == ') ' else terminals

        return f'({item.rule.lhs} {terminals}) '

    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence (gaps, not tokens)
        self.cols = [Agenda() for _ in range(self.n_cols)]

        # Start looking for ROOT at position 0
        # This just predicts all the ROOT rules in col 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        #
        # The iterator over numbered columns is `enumerate(self.cols)`.
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols), total=len(self.cols), disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:  # while agenda isn't empty
                item = column.pop()  # dequeue the next unprocessed item
                next = item.next_symbol()
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next) and (i, next) not in self.predicted_cats:
                    # Predict the nonterminal after the dot
                    # i.e. add every rule that has the nonterminal on LHS
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        self.predicted_cats[(position, nonterminal)] = True
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position, weight=0, backpointers=[])
            self.cols[position].push(new_item)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position,
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced(item.next_symbol(), 0)
            self.cols[position + 1].push(new_item)
            log.debug(f"\tScanned to get: {new_item} in column {position + 1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        object.__setattr__(item, "weight", item.weight + item.rule.weight)
        mid = item.start_position  # start position of this item = end position of item to its left
        customer_col = self.cols[mid]

        for customer in customer_col.all():  # could you eliminate this inefficient linear search?
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced(item, item.weight)  # backpointer goes to the item

                # Now push the customer with dot advanced after potentially updating backpointers
                self.cols[position].push(new_item)
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                # log.debug(f"\tweight: {self.weights[item]} + {self.weights[customer]} = {self.weights[new_item]}")
                self.profile["ATTACH"] += 1


class Agenda:
    def __init__(self) -> None:
        self._items: List[Item] = []  # list of all items that were *ever* pushed
        self._next = 0  # index of first item that has not yet been popped

        # stores index of an item if it has been pushed before
        # keys: (Rule, start pos, dot pos)
        # vals: (index in self._items, item weight)
        self._index: Dict[(Rule, int, int), (int, float)] = {}
        self.priority_q: List[int] = []  # list of indices of better constituents to reprocess

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        item_info = item.rule, item.start_position, item.dot_position
        if item_info not in self._index:  # O(1) lookup in hash table
            self._items.append(item)
            self._index[item_info] = (len(self._items) - 1, item.weight)
        # if item is alrdy in the column but item has better weight than the one in the column:
        # delete the old one completely, add item to end of self._items to process later
        elif item.weight < self._index[item_info][1]:
            idx = self._index[item_info][0]
            fat_item = self._items[idx]
            object.__setattr__(fat_item, "weight", item.weight)
            object.__setattr__(fat_item, "backpointers", item.backpointers)
            self._index[item_info] = (idx, item.weight)
            # fat_item.weight = item.weight
            # fat_item.backpointers = item.backpointers
            self.priority_q.append(idx)

            # self.junk(idx)  # completely remove fat item from push history of agenda
            # self._items.append(item)
            # self._index[item_info] = (len(self._items) - 1, item.weight)

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self) == 0:
            raise IndexError
        # if there is sth in priority queue, pop that
        elif len(self.priority_q) != 0:
            return self._items[self.priority_q.pop(0)]
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if
        they've already been popped."""
        return self._items

    # def junk(self, idx: int):
    #     """Given index of item to remove from consideration, replace
    #     the object at that index with a junk value"""
    #     self._items[idx] = Item()

    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

    @property
    def items(self):
        return self._items


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol,
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}  # maps each LHS to the list of rules that expand it

        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule.
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"

    def contains_any(self, words: List):
        for word in words:
            if word in self.rhs:
                return True
        return False

    def has_terminal(self, grammar: Grammar) -> bool:
        for symbol in self.rhs:
            if not grammar.is_nonterminal(symbol):
                return True
        return False


# We particularly want items to be immutable, since they will be hashed and
# used as keys in a dictionary (for duplicate detection).
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    weight: float
    backpointers: List

    # Each rule that formed as the result of a dot advancing has
    # backpointer to the completed item and the customer's backpointer (?)

    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for
    # debugging purposes if you wanted.

    # def pointers_updated(self, p1, p2):
    #     return Item(rule=self.rule, dot_position=self.dot_position,
    #                 start_position=self.start_position, ci_pointer=p1, cust_pointer=p2)

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def prev_symbol(self) -> Optional[str]:
        """What's the symbol before the dot? (terminal, non-terminal, or None)"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == 0:
            return None
        else:
            return self.rule.rhs[self.dot_position - 1]

    def with_dot_advanced(self, goods, goods_weight) -> Item:
        # goods is the completed item that attaches to the customer
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1,
                    start_position=self.start_position,
                    weight=self.weight + goods_weight,
                    backpointers=self.backpointers + [goods])

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        # matches notation on slides
        # return f"({self.start_position}, {self.weight}, {dotted_rule}, {self.backpointers})"
        return f"({self.start_position}, {dotted_rule})"


def preprocess_grammar(grammar, tokens):
    """fn to remove all rules in grammar that
    do not consider tokens of the sentence"""
    current_grammar = deepcopy(grammar)
    for lhs, exp_list in grammar._expansions.items():
        idx_del = []
        for exp_idx in range(len(exp_list)):
            rule = exp_list[exp_idx]
            if rule.has_terminal(grammar):
                if not rule.contains_any(tokens):
                    # delete this rule from the deep copy grammar
                    idx_del.append(exp_idx)
        for idx in sorted(idx_del, reverse=True):
            del current_grammar._expansions[lhs][idx]

    return current_grammar


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.verbose)  # Set logging level appropriately

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # if sentence is not blank, ie skip blank lines
                # analyze the sentence
                log.debug("=" * 70)
                log.debug(f"Parsing sentence: {sentence}")
                tokens = sentence.split()
                processed_grammar = preprocess_grammar(grammar, tokens)
                chart = EarleyChart(tokens, processed_grammar, progress=args.progress)

                if chart.accepted():
                    # find the lowest weight ROOT expansion with find_best
                    # print parse and weight
                    sent, weight = chart.find_best_sentence()
                    parsed_sent = chart.print_item(sent)
                    print(parsed_sent)
                    print(weight)
                else:
                    print("NONE")

                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)  # run tests
    main()
