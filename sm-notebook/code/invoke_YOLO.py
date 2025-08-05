infer_start_time = time.time()

# Read the image into a numpy array
orig_image = cv2.imread("images-test/a.jpg")

# Conver the array into jpeg
jpeg = cv2.imencode(".jpg", orig_image)[1]
# Serialize the jpg using base 64
payload = base64.b64encode(jpeg).decode("utf-8")

conf = 0.85
iou = 0.8
payload = f"{payload},{conf},{iou}"

runtime = boto3.client("runtime.sagemaker")
response = runtime.invoke_endpoint(
    EndpointName="", ContentType="text/csv", Body=payload
)

response_body = response["Body"].read()
result = json.loads(response_body.decode("ascii"))

infer_end_time = time.time()

print(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")

print(result)
