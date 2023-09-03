################################################################################
## Set up Personal Access Token and Access information
##     - This will be used by every Clarifai API call 
################################################################################
## Specify the Authorization key.  This should be changed to your Personal Access Token.
## Example: metadata = (('authorization', 'Key 123457612345678'),) 
##
## A UserAppIDSet object is needed for most rpc calls.  This object contains
## two pieces of information: the user id and the app id.  Both of these are
## specified as string values.
##
##     'user_id' : This is your user id
##     'app_id'  : This is the app id which contains the model of interest
# end of first trial

###################################################################################
# In this section, we set the user authentication, app and model IDs, and the URL
# of the image we want as an input. Change these strings to run your own example.
###################################################################################
import os
import pickle
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
# Your PAT (Personal Access Token) can be found in the portal under Authentification
# Your PAT (Personal Access Token) can be found in the portal under Authentification
PAT = 'd04d0e7225924adeb154a90eb136471d'
# Specify the correct user_id/app_id pairings
# Since you're making inferences outside your app's scope
USER_ID = 'clarifai'
APP_ID = 'main'
# Change these to whatever model and image URL you want to use
MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa9ca48295b37401f8af92ad1af0d91d'
############################################################################
# YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
############################################################################


def get_tags_from_photo(file_bytes):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    output = post_model_outputs_response.outputs[0]
    predicted_tags = [concept.name for concept in output.data.concepts]
    predicted_tags_value = [f"{concept.name}, {concept.value}" for concept in output.data.concepts]
    comma_separated_strings = ",".join(predicted_tags_value)
    return predicted_tags, comma_separated_strings