QUALITY_CHOICES = {"auto", "low", "medium", "high"}
FORMAT_CHOICES = {"png", "jpeg", "jpg", "webp"}
BACKGROUND_CHOICES = {"auto", "opaque", "transparent"}
MODERATION_CHOICES = {"auto", "low"}
ACTION_CHOICES = {"auto", "generate", "edit"}
XAI_RESPONSE_FORMAT_CHOICES = {"url", "b64_json"}
GEMINI_IMAGE_SIZES = {"512", "1K", "2K", "4K"}
GEMINI_ASPECT_RATIOS = {
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
}
XAI_ASPECT_RATIOS = {
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "2:1",
    "1:2",
    "19.5:9",
    "9:19.5",
    "20:9",
    "9:20",
}
XAI_RESOLUTIONS = {"1k", "2k"}
POPULAR_SIZE_CHOICES = {
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
}
