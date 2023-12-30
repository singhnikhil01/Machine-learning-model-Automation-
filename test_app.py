from autoML.utilities.numbers import add_numbers
# from autoML.logging import logger


def test_app():
    result = add_numbers(1, 2)
    assert result == 3

    # # Adding logging information
    # if result == 3:
    #     logger.info("Test passed: sum(1, 2) equals 3")
    # else:
    #     logger.error(f"Test failed: sum(1, 2) returned {result}, expected 3")
