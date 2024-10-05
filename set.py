from itertools import combinations

""" The file is an adaptation on the game called Set. It gives how many possible 'sets' we can find with a given hand,
where our hand is a 12 cards from 0 to 2."""

def count_sets(cards):
    """Return the number of sets in the provided Set hand.

    Parameters:
        cards (list(str)) a list of twelve cards as 4-bit integers in
        base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
        (int) The number of sets in the hand.
    Raises:
        ValueError: if the list does not contain a valid Set hand, meaning
            - there are not exactly 12 cards,
            - the cards are not all unique,
            - one or more cards does not have exactly 4 digits, or
            - one or more cards has a character other than 0, 1, or 2.
    """
    # Assure we have 12 cards
    if len(cards) != 12:
        raise ValueError("Length of cards must be 12")
    
    # Assure cards are all unique
    if len(cards) != len(set(cards)):
        raise ValueError("All cards must be unique.")

    # Assure all cards have exactly 4 digits, and assure all characters are 0,1, or 2
    for card in cards:
        if len(card) != 4:
            raise ValueError("Cards must be of length 4")
        for character in card:
            if character not in "012":
                raise ValueError("Character must be 0,1, or 2")

    # Check to see how many combinations of sets we can make
    count = 0                                     
    for i in combinations(cards, 3):
        if is_set(*i):                   # Use asterick to pass it as a three arguments to is_set
            count += 1 
    return count

def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.

    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
            For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b,
            and c are either the same or all different for i=1,2,3,4.
        False if a, b, and c do not form a set.
    """
    # Determine if three cards constitutes a set
    for i in range(4):
        if int(a[i]) + int(b[i]) + int(c[i]) not in {0, 3, 6}: # Check to make sure our sum is a multiple of 3
            return False
    return True

# if __name__ == "__main__":
#     hand = ["1022", "1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0200", "2110", "1020"]
#     print(count_sets(hand))