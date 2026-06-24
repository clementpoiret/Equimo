"""Run a small VisionTransformer for logits and intermediate features."""

import jax.random as jr

import equimo.vision.models as em


def main() -> None:
    key = jr.PRNGKey(0)
    model_key, image_key, logits_key, features_key = jr.split(key, 4)

    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=10,
        key=model_key,
    )
    image = jr.normal(image_key, (3, 64, 64))

    logits = model(image, key=logits_key, inference=True)
    features = model.features(image, key=features_key, inference=True)

    print("logits:", logits.shape)
    print("features:", features.shape)


if __name__ == "__main__":
    main()
