import logging

def setup_logger(log_file='app.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info('Logging setup complete.')
