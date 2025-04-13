import time


def wait_for_file_lines(file_path, timeout=20, interval=0.1):
    """
    Waits until its contents (lines) can be read.

    Parameters:
        file_path (str): The path of the file to be read.
        timeout (int, optional): The maximum number of seconds to wait before timing out.
        interval (float, optional): The interval in seconds between checks.
    """
    start_time = time.time()

    while True:
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            # Return the lines if the list is not empty
            if lines:
                return True
        except Exception:
            # If there's an error (such as the file being in use),
            # ignore and retry after the interval
            pass

        # Check if the timeout has been exceeded
        if time.time() - start_time > timeout:
            return False

        # Wait for the specified interval before retrying
        time.sleep(interval)
