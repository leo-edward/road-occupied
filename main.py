import json
from __init__ import *
from src import pulsar_client, post_processing
from concurrent.futures import ThreadPoolExecutor

args = None

def run_inference(message):
    image_process = post_processing.Image_detect()
    result = image_process.image_process(message)
    if result is not None:
        # logger.info("result is: {}, sending results...".format(result))
        data_send = json.dumps(result)
        client_sent = pulsar_client.ResultsProcess(producer_topic)
        client_sent.send_message(data_send)
    else:
        logger.info("result is None.")


if __name__ == "__main__":
    logger.info("START ROAD OCCUPIED ALGORITHM")
    threadPool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="road_occupied_")
    logger.info("Starting pulsar ...")
    try:
        client = pulsar_client.ResultsProcess(consumer_topic)
    except Exception as err:
        logger.error("An error has occurred when creat connection to pulsar topic {}".format(consumer_topic))
        logger.error(err)
        exit(1)
    while True:
        # Get message from pulsar
        message = client.get_results()
        ThreadPoolExecutor.submit(threadPool, run_inference, message)
        # run_inference(message)

    client.close()
