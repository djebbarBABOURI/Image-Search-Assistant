from mcp.server.mcpserver import MCPServer
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pets-server")

mcp = MCPServer(name="pets-server", version="1.0.0", description="Oxford Pets MCP Server")


@mcp.tool()
def classify(image_id: str, top_k: int = 3) -> dict:
    """Predict the breed of an image and give the top probabilities."""


@mcp.tool()
def find_similar(image_id: str, k: int = 5) -> dict:
    """Find the k images closest to a given image in the embedding space."""


@mcp.tool()
def filter_images(label: str, min_confidence: float = 0.0, limit: int = 10) -> dict:
    """List the images PREDICTED as a given breed with a minimum confidence."""



@mcp.tool()
def count_by(label: str = "", min_confidence: float = 0.0) -> dict:
    """Count images by predicted breed."""


@mcp.resource("pets://classes")
def classes_resource() -> str:
    """Expose the 37 valid breed names, one per line, with their species."""


@mcp.resource("pets://stats")
def stats_resource() -> str:
    """Expose general statistics about the collection and the predictions."""


@mcp.resource("pets://metrics")
def metrics_resource() -> str:
    """Expose the evaluation metrics of the trained model."""


@mcp.resource("pets://image/{image_id}")
def image_resource(image_id: str) -> str:
    """Expose the summary of one image of the collection."""



@mcp.prompt()
def find_look_alikes(image_id: str) -> str:
    """Build a prompt asking for the neighbours of an image and their breeds."""
    return (
        f"Trouve les 5 images les plus proches de {image_id} avec find_similar. "
        f"Puis appelle classify sur {image_id} et sur chaque voisin. "
        "Présente un tableau (image, race prédite, confiance) et dis si les "
        "voisins ont la même race que l'image de départ."
    )



if __name__ == "__main__":
    mcp.run()