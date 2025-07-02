# FastVideo Documentation

This directory contains the comprehensive documentation for FastVideo, a high-performance framework for training and inference of video diffusion models.

## Building the Documentation

### Prerequisites

1. Install documentation dependencies:
```bash
pip install -r docs/requirements-docs.txt
```

2. Ensure FastVideo is installed in your environment:
```bash
pip install -e .
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be available in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Live Development

For development with auto-reload:

```bash
cd docs
make livehtml
```

This will start a local server with auto-reload functionality. The documentation will be available at `http://localhost:8000` and will automatically rebuild when you make changes to the source files.

### Other Build Targets

- **Clean build artifacts**: `make clean`
- **Check links**: `make linkcheck`
- **Coverage report**: `make coverage`
- **Run doctests**: `make doctest`
- **PDF output**: `make latexpdf` (requires LaTeX installation)

## Documentation Structure

### Main Files

- `index.rst` - Main documentation index
- `training_api.rst` - Comprehensive Training APIs reference
- `data_preprocess.md` - Data preprocessing guide
- `conf.py` - Sphinx configuration
- `Makefile` - Build automation

### Directory Structure

```
docs/
├── README.md              # This file
├── index.rst             # Main documentation index
├── training_api.rst      # Training APIs documentation
├── data_preprocess.md    # Data preprocessing guide
├── conf.py               # Sphinx configuration
├── Makefile              # Build automation
├── requirements-docs.txt # Documentation dependencies
├── _static/              # Static files (CSS, images)
│   └── custom.css        # Custom styling
├── _templates/           # Custom templates
└── _build/              # Generated documentation (excluded from git)
```

## Documentation Content

### Training APIs Reference (`training_api.rst`)

Comprehensive documentation covering:

- **Core Training Scripts**: Main training, distillation, adversarial training
- **V1 API Components**: Inference engine, models, pipelines, distributed training
- **Model Components**: DiT models, VAE models, text encoders, schedulers
- **Pipeline System**: Modular pipeline stages and composition
- **Distributed Training**: FSDP, sequence parallelism, tensor parallelism
- **Parameter Management**: Advanced parameter loading and sharding
- **CLI Interface**: Command-line tools and utilities
- **Best Practices**: Training tips, performance optimization, troubleshooting
- **Configuration Examples**: Ready-to-use configuration templates

### Key Features Documented

1. **Multiple Training Modes**:
   - Standard flow matching training
   - Knowledge distillation for faster inference
   - LoRA fine-tuning for parameter efficiency
   - Adversarial training for enhanced quality

2. **Model Architectures**:
   - Mochi: High-quality video generation
   - Hunyuan Video: Large-scale video models
   - Custom model support

3. **Advanced Features**:
   - Multi-GPU distributed training
   - Sequence parallelism for long videos
   - Mixed precision training
   - Gradient checkpointing
   - CPU offloading

4. **Performance Optimizations**:
   - Efficient attention mechanisms
   - Optimized data loading
   - Memory-efficient training
   - TF32 acceleration

## Writing Documentation

### Style Guidelines

1. **Use reStructuredText (.rst) for API documentation**
2. **Use Markdown (.md) for guides and tutorials**
3. **Include comprehensive docstrings in code**
4. **Provide working code examples**
5. **Document all parameters and return values**

### Adding New Documentation

1. Create new `.rst` or `.md` files in the `docs/` directory
2. Add the file to the `toctree` in `index.rst`
3. Follow the existing documentation structure and style
4. Include relevant cross-references and links
5. Test the build with `make html`

### API Documentation

The documentation uses Sphinx autodoc to automatically generate API documentation from docstrings. Key directives:

- `.. py:module::` - Document a module
- `.. py:class::` - Document a class
- `.. py:function::` - Document a function
- `.. py:method::` - Document a method

### Code Examples

Include working code examples:

```rst
.. code-block:: python

   from fastvideo.v1.inference_engine import InferenceEngine
   
   # Create engine
   engine = InferenceEngine.create_engine(args)
   result = engine.run("A cat playing", args)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure FastVideo is installed (`pip install -e .`)
2. **Missing Dependencies**: Install docs requirements (`pip install -r docs/requirements-docs.txt`)
3. **Build Failures**: Check syntax with `make clean && make html`
4. **Missing Modules**: Modules are mocked in `conf.py` autodoc_mock_imports

### Getting Help

- Check existing documentation build
- Review Sphinx documentation: https://www.sphinx-doc.org/
- Check reStructuredText primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

## Contributing

When contributing to documentation:

1. Follow the existing style and structure
2. Test your changes with `make html`
3. Ensure all code examples work
4. Update this README if adding new sections
5. Include relevant cross-references

The documentation is built automatically in CI/CD and deployed to the project website.