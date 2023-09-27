docker run -d \
    -v /data2/algorithm-controler/algorithm-logs:/app/log \
    zhentian/road-occupied-client:1.0 \
    python3 main.py --config {\"algorithm_params\":{\"triton_url\":\"192.168.113.55:40001\"},\"algorithm\":\"road_occupied\",\"consumer_topic\":\"Graphinfo_people_gather\",\"producer_topic\":\"capture_scene\",\"pulsar_address\":\"pulsar://192.168.113.71:6650\",\"log\":{\"log_path\":\"/app/log\",\"log_file\":\"capture_face-algo_input_face_101_client\",\"log_rotation\":\"1 week\",\"log_retention\":\"3 week\"}}