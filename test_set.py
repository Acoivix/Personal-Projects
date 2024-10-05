import set
import pytest

"""Tests that our Set game works properly. Ensure's 100 percent coverage."""    
def test_count_sets():
    # Tests to make sure we have 12 cards exactly
    with pytest.raises(ValueError):
       set.count_sets(["1022", "1122"])

    # Tests that cards are all unique
    with pytest.raises(ValueError):
        hand_with_duplicate = ["1022", "1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0200", "2110", "0020"]
        set.count_sets(hand_with_duplicate)
    
    # Tests that cards all have exactly 4 digits
    with pytest.raises(ValueError):
        hand_with_3digits = ["022", "1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0200", "2110", "1020"]
        set.count_sets(hand_with_3digits)
    
    # Tests that all cards have character 0,1, or 2
    with pytest.raises(ValueError):
        hand_with_other_digit = ["3022", "1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0200", "2110", "1020"]
        set.count_sets(hand_with_other_digit)

    # Test that our function returns the right amount of sets
    valid_hand = ["1022", "1122", "0100", "2021", "0010", "2201", "2111", "0020", "1102", "0200", "2110", "1020"]
    expected_number_of_sets = 6
    assert set.count_sets(valid_hand) == expected_number_of_sets
    
    
def test_is_set():
    # Test that all similar is a set
    assert set.is_set("0000", "0000", "0000") == True

    # Test that all different is a set
    assert set.is_set("0102","1210", "2021") == True

    # Test that some same, some different is a set
    assert set.is_set("0111", "0222", "0000") == True

    # Test that a non set returns False
    assert set.is_set("1022", "1122", "1020") == False


    
