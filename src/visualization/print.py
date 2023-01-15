def print_separator(title: str = None):
    """
    Print a separation line to stdout

    Args:
        title (str, optional): Optional title to display in the middle of the separator. Defaults to None.
    """
    
    SEPARATOR_LENGTH = 40
    SEPARATOR = "#"
    
    if title is not None:
    
        additional_separators = (SEPARATOR_LENGTH - len(title) - 2)
        semi_separator_len = additional_separators // 2
        extra_separator = additional_separators % 2
        print(SEPARATOR * semi_separator_len + " " + title + " " + SEPARATOR * (semi_separator_len + extra_separator))
    
    else: 
        print(SEPARATOR * SEPARATOR_LENGTH + "\n")