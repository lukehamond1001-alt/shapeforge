# Imagen Apex Examples

These chair models were generated using the [Imagen Apex](https://github.com/lukehamond1001-alt/imagen-apex) text-to-3D pipeline.

## Files

| File | Description | Size |
|------|-------------|------|
| `chair_imagen_apex_01.ply` | Minimalist wooden chair | ~16MB |
| `chair_imagen_apex_02.ply` | Modern chair design | ~16MB |

## Generation Process

1. **Text Prompt** → Gemini Pro generates a 2D concept image
2. **2D Image** → SAM 3D converts to Gaussian splat point cloud
3. **Export** → PLY file with ~300K+ points

## Viewing

Open these files in any 3D viewer:
- [MeshLab](https://www.meshlab.net/) (free)
- [Blender](https://www.blender.org/) (free)
- [CloudCompare](https://www.cloudcompare.org/) (free)

## Comparison with ShapeForge

These models represent the **inference** approach — using pretrained models (Gemini + SAM 3D) to generate from text prompts.

ShapeForge demonstrates the **training** approach — fine-tuning a model on domain-specific data (ShapeNet chairs) for specialized generation.

Both approaches are valuable for different use cases!
