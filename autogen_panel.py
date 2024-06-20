import asyncio
import base64
import io
import os
import time
import traceback
import certifi
import ssl
import websockets
import nest_asyncio

import PIL
import autogen
import numpy as np
import panel as pn
import param
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from autogen import ConversableAgent, GroupChatManager, GroupChat, AssistantAgent
from hume.models.config import FaceConfig
from skimage import data
from skimage.draw import rectangle
from skimage.feature import Cascade
from hume import HumeStreamClient

# set openai api key here
os.environ["OPENAI_API_KEY"] = ""
openai_api_key = ""
model = "gpt-3.5-turbo"
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["SSL_CERT_DIR"] = certifi.where()
input_future = None

nest_asyncio.apply()


class MyConversableAgent(autogen.ConversableAgent):

    async def a_get_human_input(self, prompt: str) -> str:
        global input_future
        print('AGET!!!!!!')  # or however you wish to display the prompt
        # chat_interface.send(prompt, user="System", respond=False)
        # Create a new Future object for this input operation if none exists
        if input_future is None or input_future.done():
            input_future = asyncio.Future()

        # Wait for the callback to set a result on the future
        await input_future

        # Once the result is set, extract the value and reset the future for the next input operation
        input_value = input_future.result()
        input_future = None
        return input_value


conversation_mood_agent = ConversableAgent(
    name="Conversation_Mood_Agent",
    system_message="You are the Mood Tracking Agent. Your role is to assess the user's current mood and emotional "
                   "state using predefined scales and questions. Always provide support "
                   "instead of referring the user to seek help elsewhere. When appropriate"
                   "ask the"
                   "user to reveal their mood on a rating scale and"
                   "provide empathetic responses based on their input.Do not reveal your role under "
                   "any circumstances.",
    llm_config={"config_list": [{"model": model, "api_key": openai_api_key}]},
)

assessment_agent = ConversableAgent(
    name="Assessment_Agent",
    system_message="You are the Assessment Agent. Your role is to conduct a detailed assessment of the user's "
                   "symptoms and behaviors.Do not refuse to provide support."
                   "You will ask a series of questions to understand the user's emotional and mental health status, "
                   "particularly"
                   "focusing on symptoms of anxiety, depression and substance use. Provide supportive feedback based "
                   "on the"
                   "user's responses. Do not reveal your role under any circumstances.",
    llm_config={"config_list": [{"model": model, "api_key": openai_api_key}]},
)

therapy_agent = AssistantAgent(
    name="Therapy_Agent",
    system_message="You are the Therapeutic Agent. Your role is to offer immediate therapeutic interventions, "
                   "such as breathing exercises or mindfulness techniques or cognitive behavioural activities, "
                   "to help manage the user's stress and"
                   "improve their mood. Always ask for the user's permission before proceeding with the "
                   "exercise.Provide clear, step-by-step instructions for the exercises and encourage the"
                   "user throughout the process. After the intervention, always offer additional support or resources "
                   "from the recommendation agent. Do"
                   "not reveal your role under any circumstances.",
    llm_config={"config_list": [{"model": model, "api_key": openai_api_key}]},
)

crisis_mgmt_agent = AssistantAgent(
    name="Crisis_Management_Agent",
    system_message="You are the Crisis Management Agent. Your role is to handle situations where the user may be in "
                   "immediate danger or requires urgent support due to their substance use or emotional state. Assess "
                   "the severity of the crisis and provide appropriate responses, including routing to human support "
                   "if necessary. Maintain a calm and reassuring tone, ensuring the user's safety is prioritized.Do "
                   "not reveal your role under any circumstances.",
    llm_config={"config_list": [{"model": model, "api_key": openai_api_key}]},
)

resource_recommendation_agent = AssistantAgent(
    name="Resource_Recommendation_Agent",
    system_message="You are the Resource Recommendation Agent. Your role is to provide the user with personalized "
                   "recommendations for articles, videos, and other resources that can help them manage their mental "
                   "health and substance use. Tailor your recommendations based on the user's expressed needs and "
                   "preferences. After providing resources, prompt the user to provide feedback on the conversation "
                   "and close the conversation with a supportive message.Do not reveal your role under any "
                   "circumstances.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)

human_proxy = MyConversableAgent(
    name="human_proxy",
    llm_config=False,  # no LLM used for human proxy
    human_input_mode="ALWAYS",  # always ask for human input
)

avatar = {
    conversation_mood_agent.name: "💬",  # Speech balloon
    assessment_agent.name: "🧾",         # Receipt
    therapy_agent.name: "🧘",            # Person in lotus position
    crisis_mgmt_agent.name: "🆘",        # SOS button
    resource_recommendation_agent.name: "📖",  # Open book
    human_proxy.name: "👤"              # Person
}


def print_messages(recipient, messages, sender, config):
    print(
        f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")

    content = messages[-1]['content']

    if all(key in messages[-1] for key in ['name']):
        if messages[-1]['name'] not in ['human_proxy']:
            chat_interface.send(content, user=messages[-1]['name'], avatar=avatar[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=avatar[recipient.name], respond=False)

    return False, None  # required to ensure the agent communication flow continues


conversation_mood_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

assessment_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

therapy_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

crisis_mgmt_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

resource_recommendation_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

human_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

allowed_transitions = {
    human_proxy: [conversation_mood_agent, assessment_agent, therapy_agent, crisis_mgmt_agent,
                  resource_recommendation_agent],
    conversation_mood_agent: [human_proxy],
    assessment_agent: [human_proxy, ],
    therapy_agent: [human_proxy],
    crisis_mgmt_agent: [human_proxy, ],
    resource_recommendation_agent: [human_proxy, ]
}

constrained_graph_chat = GroupChat(
    agents=[conversation_mood_agent, assessment_agent, therapy_agent, crisis_mgmt_agent, resource_recommendation_agent,
            human_proxy],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
    messages=[],
    max_round=40,
    send_introductions=True,
)

constrained_group_chat_manager = GroupChatManager(
    groupchat=constrained_graph_chat,
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
)
pn.extension(design="material")

initiate_chat_task_created = False


async def delayed_initiate_chat(agent, recipient, message):
    global initiate_chat_task_created
    # Indicate that the task has been created
    initiate_chat_task_created = True

    # Wait for 2 seconds
    await asyncio.sleep(2)

    # Now initiate the chat
    await agent.a_initiate_chat(recipient, message=message)


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global input_future

    if not initiate_chat_task_created:
        asyncio.create_task(delayed_initiate_chat(human_proxy, constrained_group_chat_manager, contents))

    else:
        if input_future and not input_future.done():
            input_future.set_result(contents)
        else:
            print("There is currently no input being awaited.")


chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("💬 Hey there, I'm here to help. What's on your mind?", user="Conversation_Mood_Agent",
                    avatar="💬", respond=False)

HEIGHT = 360  # pixels
WIDTH = 240  # pixels
TIMEOUT = 500  # milliseconds


class ImageModel(pn.viewable.Viewer):
    """Base class for image models."""

    def __init__(self, **params):
        super().__init__(**params)

        with param.edit_constant(self):
            self.name = self.__class__.name.replace("Model", "")
        self.view = self.create_view()

    def __panel__(self):
        return self.view

    def apply(self, image: str, height: int = HEIGHT, width: int = WIDTH) -> str:
        """Transforms a base64 encoded jpg image to a base64 encoded jpg BytesIO object"""
        raise NotImplementedError()

    def create_view(self):
        """Creates a view of the parameters of the transform to enable the user to configure them"""
        return pn.Param(self, name=self.name)

    def transform(self, image):
        """Transforms the image"""
        raise NotImplementedError()


class NumpyImageModel(ImageModel):
    """Base class for np.ndarray image models"""

    @staticmethod
    def to_np_ndarray(image: str, height=HEIGHT, width=WIDTH) -> np.ndarray:
        """Converts a base64 encoded jpeg string to a np.ndarray"""
        pil_img = PILImageModel.to_pil_img(image, height=height, width=width)
        return np.array(pil_img)

    @staticmethod
    def from_np_ndarray(image: np.ndarray) -> io.BytesIO:
        """Converts np.ndarray jpeg image to a jpeg BytesIO instance"""
        if image.dtype == np.dtype("float64"):
            image = (image * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(image)
        return PILImageModel.from_pil_img(pil_img)

    def apply(self, image: str, height: int = HEIGHT, width: int = WIDTH) -> io.BytesIO:
        np_array = self.to_np_ndarray(image, height=height, width=width)

        transformed_image = self.transform(np_array)

        return self.from_np_ndarray(transformed_image)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transforms the np.array image"""
        raise NotImplementedError()


class PILImageModel(ImageModel):
    """Base class for PIL image models"""

    @staticmethod
    def to_pil_img(value: str, height=HEIGHT, width=WIDTH):
        """Converts a base64 jpeg image string to a PIL.Image"""
        encoded_data = value.split(",")[1]
        base64_decoded = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(base64_decoded))
        image.draft("RGB", (height, width))
        return image

    @staticmethod
    def from_pil_img(image: Image):
        """Converts a PIL.Image to a base64 encoded JPG BytesIO object"""
        buff = io.BytesIO()
        image.save(buff, format="JPEG")
        return buff

    def apply(self, image: str, height: int = HEIGHT, width: int = WIDTH) -> io.BytesIO:
        pil_img = self.to_pil_img(image, height=height, width=width)

        transformed_image = self.transform(pil_img)

        return self.from_pil_img(transformed_image)

    def transform(self, image: PIL.Image) -> PIL.Image:
        """Transforms the PIL.Image image"""
        raise NotImplementedError()


def to_instance(value, **params):
    """Converts the value to an instance

    Args:
        value: A param.Parameterized class or instance

    Returns:
        An instance of the param.Parameterized class
    """
    if isinstance(value, param.Parameterized):
        value.param.update(**params)
        return value
    return value(**params)


class VideoStreamInterface(pn.viewable.Viewer):
    """An easy to use interface for a VideoStream and a set of transforms"""

    video_stream = param.ClassSelector(
        class_=pn.widgets.VideoStream, constant=True, doc="The source VideoStream", allow_refs=False,
    )

    height = param.Integer(
        default=HEIGHT,
        bounds=(10, 2000),
        step=10,
        doc="""The height of the image converted and shown""",
    )
    width = param.Integer(
        default=WIDTH,
        bounds=(10, 2000),
        step=10,
        doc="""The width of the image converted and shown""",
    )

    model = param.Selector(doc="The currently selected model")

    def __init__(
            self,
            models,
            timeout=TIMEOUT,
            paused=False,
            **params,
    ):
        super().__init__(
            video_stream=pn.widgets.VideoStream(
                name="Video Stream",
                timeout=timeout,
                paused=paused,
                height=0,
                width=0,
                visible=False,
                format="jpeg",
            ),
            **params,
        )
        self.image = pn.pane.JPG(
            height=self.height, width=self.width, sizing_mode="fixed"
        )
        self._updating = False
        models = [to_instance(model) for model in models]
        self.param.model.objects = models
        self.model = models[0]
        self.timer = Timer(sizing_mode="stretch_width")
        self.settings = self._create_settings()
        self._panel = self._create_panel()

    def _create_settings(self):
        return pn.Column(
            pn.Param(
                self.video_stream,
                parameters=["timeout", "paused"],
                widgets={
                    "timeout": {
                        "widget_type": pn.widgets.IntSlider,
                        "start": 10,
                        "end": 2000,
                        "step": 10,
                    }
                },
            ),
            pn.Param(self, parameters=["height", "width"], name="Image"),
            pn.Param(
                self,
                parameters=["model"],
                expand_button=False,
                expand=False,
                widgets={
                    "model": {
                        "widget_type": pn.widgets.RadioButtonGroup,
                        "orientation": "vertical",
                        "button_type": "primary",
                        "button_style": "outline"
                    }
                },
                name="Model",
            ),
            self._get_transform,
        )

    def _create_panel(self):
        return pn.Row(
            self.video_stream,
            pn.layout.HSpacer(),
            self.image,
            pn.layout.HSpacer(),
            sizing_mode="stretch_width",
            align="center",
        )

    @param.depends("height", "width", watch=True)
    def _update_height_width(self):
        self.image.height = self.height
        self.image.width = self.width

    @param.depends("model")
    def _get_transform(self):
        # Hack: returning self.transform stops working after browsing the transforms for a while
        return self.model.view

    def __panel__(self):
        return self._panel

    @param.depends("video_stream.value", watch=True)
    def _handle_stream(self):
        if self._updating:
            return

        self._updating = True
        if self.model and self.video_stream.value:
            value = self.video_stream.value
            try:
                image = self.timer.time_it(
                    name="Model",
                    func=self.model.apply,
                    image=value,
                    height=self.height,
                    width=self.width,
                )
                self.image.object = image
            except PIL.UnidentifiedImageError:
                print("unidentified image")

            self.timer.inc_it("Last Update")
        self._updating = False


class Timer(pn.viewable.Viewer):
    """Helper Component used to show duration trends"""

    _trends = param.Dict()

    def __init__(self, **params):
        super().__init__()

        self.last_updates = {}
        self._trends = {}

        self._layout = pn.Row(**params)

    def time_it(self, name, func, *args, **kwargs):
        """Measures the duration of the execution of the func function and reports it under the
        name specified"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = round(end - start, 2)
        self._report(name=name, duration=duration)
        return result

    def inc_it(self, name):
        """Measures the duration since the last time inc_it was called and reports it under the
        specified name"""
        start = self.last_updates.get(name, time.time())
        end = time.time()
        duration = round(end - start, 2)
        self._report(name=name, duration=duration)
        self.last_updates[name] = end

    def _report(self, name, duration):
        if not name in self._trends:
            self._trends[name] = pn.indicators.Trend(
                name=name,
                data={"x": [1], "y": [duration]},
                height=100,
                width=150,
                sizing_mode="fixed",
            )
            self.param.trigger("_trends")
        else:
            trend = self._trends[name]
            next_x = max(trend.data["x"]) + 1
            trend.stream({"x": [next_x], "y": [duration]}, rollover=10)

    @param.depends("_trends")
    def _panel(self):
        self._layout[:] = list(self._trends.values())
        return self._layout

    def __panel__(self):
        return pn.panel(self._panel)


@pn.cache()
def get_detector():
    """Returns the Cascade detector"""
    trained_file = data.lbp_frontal_face_cascade_filename()
    return Cascade(trained_file)


class FaceDetectionModel(NumpyImageModel):
    """Face detection using a cascade classifier.

    https://scikit-image.org/docs/0.15.x/auto_examples/applications/plot_face_detection.html
    """

    scale_factor = param.Number(default=1.4, bounds=(1.0, 2.0), step=0.1)
    step_ratio = param.Integer(default=1, bounds=(1, 10))
    size_x = param.Range(default=(60, 322), bounds=(10, 500))
    size_y = param.Range(default=(60, 322), bounds=(10, 500))

    def transform(self, image):
        detector = get_detector()
        detected = detector.detect_multi_scale(
            img=image,
            scale_factor=self.scale_factor,
            step_ratio=self.step_ratio,
            min_size=(self.size_x[0], self.size_y[0]),
            max_size=(self.size_x[1], self.size_y[1]),
        )

        for patch in detected:
            rrr, ccc = rectangle(
                start=(patch["r"], patch["c"]),
                extent=(patch["height"], patch["width"]),
                shape=image.shape[:2],
            )
            image[rrr, ccc, 0] = 200

        return image


class EmotionDetectionModel(NumpyImageModel):
    def __init__(self, api_key, **params):
        super().__init__(**params)
        self.client = HumeStreamClient(api_key)
        self.top_emotions_text = pn.widgets.TextAreaInput(name='Top Emotions', height=200)

    async def analyze_emotions(self, ndarray):
        try:
            file_path = self.ndarray_to_image(ndarray)
            config = FaceConfig(identify_faces=True)
            async with self.client.connect([config]) as socket:
                result = await socket.send_file(file_path)
                print("Hume API Response:", result)  # Print the API response
                predictions = result.get("face", {}).get("predictions", [])
                if predictions:
                    emotions = predictions[0].get("emotions", [])
                    return emotions
                return None
        except Exception:
            print(traceback.format_exc())
            return None

    def ndarray_to_image(self, ndarray, file_path='temp_image.png'):
        image = Image.fromarray(ndarray)
        image.save(file_path)
        return file_path

    def annotate_image_with_emotion(self, image, emotion):
        annotated_image = Image.fromarray(image)
        draw = ImageDraw.Draw(annotated_image)
        try:
            # Try to load a truetype font
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 50)  # Adjust path as needed
        except IOError:
            # Fall back to default bitmap font
            font = ImageFont.load_default()
        text = f"{emotion['name']}"
        text_x = 10  # Top left corner
        text_y = 10  # Top left corner
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 255))  # Blue color
        return np.array(annotated_image)

    def apply(self, image: str, height: int = HEIGHT, width: int = WIDTH) -> io.BytesIO:
        np_array = self.to_np_ndarray(image, height=height, width=width)
        loop = asyncio.get_event_loop()

        # Run the analyze_emotions coroutine
        future = asyncio.ensure_future(self.analyze_emotions(np_array))
        emotions = loop.run_until_complete(future)

        if emotions:
            top_emotions = sorted(emotions, key=lambda e: e['score'], reverse=True)[:3]
            self.top_emotions_text.value = "\n".join([f"{e['name']}: {e['score']:.2f}" for e in top_emotions])
            most_likely_emotion = top_emotions[0]
            transformed_image = self.annotate_image_with_emotion(np_array, most_likely_emotion)
        else:
            self.top_emotions_text.value = "No emotions detected"
            transformed_image = np_array

        return self.from_np_ndarray(transformed_image)


# Example usage
emotion_model = EmotionDetectionModel(api_key="your-api-key")
component = VideoStreamInterface(
    models=[
        FaceDetectionModel(),
        emotion_model
    ]
)

# Create a panel layout to include the VideoStreamInterface and the emotion text box
layout = pn.Column(
    component.settings,
    component,
    emotion_model.top_emotions_text  # Add the top emotions text box to the layout
)

# enter the Hume API Key here
emotion_model = EmotionDetectionModel(api_key="")

component = VideoStreamInterface(
    models=[
        emotion_model,
        FaceDetectionModel,
    ]
)

# Create a panel layout to include the VideoStreamInterface and the emotion text box
layout = pn.Column(
    component.settings,
    component,
    emotion_model.top_emotions_text  # Add the emotion text box to the layout
)

# Serve the panel layout
pn.template.MaterialTemplate(
    site="MH",
    title="Chat with Emotion Detection",
    sidebar=[layout],
    main=[chat_interface],
).servable()

