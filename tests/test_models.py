import equimo.models as em
import jax.numpy as jnp
import jax.random as jr


def test_vit_inference():
    key = jr.PRNGKey(42)
    img_size = 224
    patch_size = 14

    x1 = jr.normal(key, (3, 224, 224))
    x2 = jr.normal(key, (3, 98, 98))
    mask = jr.bernoulli(key, shape=(16, 16)) * 1

    base_model = em.VisionTransformer(
        img_size=img_size,
        in_channels=3,
        dim=384,
        patch_size=patch_size,
        num_heads=[6],
        depths=[12],
        num_classes=0,
        use_mask_token=True,
        dynamic_img_size=True,
        key=key,
    )

    # Testing multiple img sizes, inference mode, and masking
    f1 = base_model.features(x1, mask=mask, inference=True, key=key)
    f2 = base_model.features(x2, inference=False, key=key)

    assert jnp.all(f1)
    assert jnp.all(f2)
