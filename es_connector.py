import base64
from elasticsearch import Elasticsearch, AuthenticationException
from loguru import logger


def get_es_engine(config: dict) -> Elasticsearch:
    """
    Initializes an Elasticsearch client using the provided configuration dictionary.

    Args:
        config (dict): Configuration details for Elasticsearch.

    Returns:
        Elasticsearch: Elasticsearch client instance.
    """
    # Extract connection details from the config
    connection = config.get('connection', {}).get('https', {})
    host_info = connection.get('hosts', [{}])[0]
    host = host_info.get('hostname')
    port = host_info.get('port')
    composed = connection.get('composed')

    # Authentication details
    username = connection.get('authentication', {}).get('username')
    password = connection.get('authentication', {}).get('password')
    use_ssl = config.get('use_ssl', True)

    # Certificate information
    certificate_authority = connection.get('certificate', {}).get('certificate_authority')
    certificate_base64 = connection.get('certificate', {}).get('certificate_base64')

    # Save the base64 certificate if it exists
    if certificate_base64:
        with open("ca_cert.pem", "wb") as cert_file:
            cert_file.write(base64.b64decode(certificate_base64))
        ca_certs = "ca_cert.pem"
    else:
        ca_certs = certificate_authority if use_ssl else None

    # Log the connection details
    logger.info(f"Connecting to Elasticsearch at {host}:{port} with user '{username}'")

    # Attempt to create the Elasticsearch client
    try:
        es = Elasticsearch(
            hosts=composed,
            basic_auth=(username, password) if username and password else None,
            verify_certs=use_ssl,
            ca_certs=ca_certs
        )

        # Test the connection
        if es.ping():
            logger.info("Connected to Elasticsearch!")
        else:
            logger.error("Could not connect to Elasticsearch.")

        return es

    except AuthenticationException:
        logger.error("Authentication failed. Please check your username and password.")
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")

    return None
