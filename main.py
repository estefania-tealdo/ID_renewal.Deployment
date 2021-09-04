from flask import Flask, request, jsonify
from model_files.passport_photo import passport_photo
# instanciate a flask object:
app = Flask("ID_segmentation")

## define the route:##
@app.route('/', methods = ["POST"])
def segment():
    
    req_data = request.get_json()
    image_req = req_data["image_b64"]
    
    import base64

    #with open("./model_files/in_pic.jpg", "wb") as fh:
    #    fh.write(base64.b64decode(image_req))



    image_segmented=passport_photo(image_req)

    #return jsonify(response)
    #response = {
    #   image_b64 = image_segmented
       #"image_b64": image_req
    #}
    return jsonify(image_b64 = image_segmented.decode('utf-8'))
    

# create a random route to test connection, GET req
#@app.route('/', methods=['GET'])
#def ping():
#   return "Pinging worked ;)"

# boiler plate code to start it:
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=9696) 






