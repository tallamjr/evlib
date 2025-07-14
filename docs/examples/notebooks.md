# Jupyter Notebooks

Our collection of Jupyter notebooks demonstrates evlib functionality with real data and practical examples.

## 📚 Available Notebooks

### 🧪 **test_data_readers_comprehensive.ipynb** ⭐ PRIORITY
**Comprehensive testing of all data loading functionality**

- **Purpose**: Validates all file format features with real data
- **Dataset**: 1,078,541 events from slider_depth dataset
- **Features Tested**:
  - Basic file loading (text, HDF5)
  - Time window filtering
  - Spatial bounds filtering
  - Polarity filtering
  - Custom column mapping
  - Header line handling
  - Round-trip save/load compatibility
- **Status**: ✅ Thoroughly tested, passes pytest --nbmake
- **Best for**: Understanding file I/O capabilities, verifying functionality

### 📊 **data_reader_demo.ipynb** ⭐ PRIORITY
**Quick introduction to event data loading**

- **Purpose**: Simple demonstration of core data loading features
- **Dataset**: Sample event data
- **Features**:
  - Basic event loading
  - Simple filtering examples
  - Quick visualization
- **Status**: ✅ Working with latest evlib
- **Best for**: New users, quick start guide

### 🎥 **evlib_event_reconstruction.ipynb**
**Event-to-video reconstruction using neural networks**

- **Purpose**: Demonstrates E2VID neural network for video reconstruction
- **Features**:
  - E2VID UNet model loading
  - Event-to-video conversion
  - Reconstruction quality analysis
- **Status**: ✅ Working (uses custom wrapper functions)
- **Best for**: Neural network applications, video reconstruction

### 🎨 **ex2_events_viz_complete.ipynb**
**Comprehensive event visualization techniques**

- **Purpose**: Educational notebook covering various visualization methods
- **Features**:
  - Multiple event representation techniques
  - Visualization exercises with solutions
  - Comparative analysis of methods
- **Status**: ✅ Working (some custom implementations)
- **Best for**: Learning visualization techniques, educational purposes

### 📈 **evlib_slider_depth.ipynb**
**Extensive dataset exploration and analysis**

- **Purpose**: Comprehensive analysis of large event dataset
- **Dataset**: Complete slider_depth dataset (1M+ events)
- **Features**:
  - Statistical analysis of event data
  - Advanced visualization techniques
  - Performance benchmarking
- **Status**: 🟡 Needs minor API updates
- **Best for**: Advanced users, dataset analysis

### 🖼️ **event_viz.ipynb**
**Detailed event visualization guide**

- **Purpose**: Comprehensive guide to event visualization
- **Features**:
  - Multiple representation methods
  - Customization techniques
  - Performance optimization
- **Status**: 🟡 Needs minor API updates
- **Best for**: Visualization experts, custom implementations

## 🚀 Running the Notebooks

### Prerequisites
```bash
pip install evlib[jupyter]
```

### Launch Jupyter
```bash
cd examples/
jupyter notebook
```

### Testing with pytest
```bash
# Test all notebooks
pytest --nbmake examples/

# Test specific notebook
pytest --nbmake examples/test_data_readers_comprehensive.ipynb
```

## 📋 Notebook Status Guide

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| ✅ | Fully working with latest evlib | None - ready to use |
| 🟡 | Minor API updates needed | Update API calls |
| ❌ | Major issues or deprecated | Not recommended |

## 🎯 Recommended Learning Path

### 1. **Beginners** - Start Here
1. `data_reader_demo.ipynb` - Learn basic data loading
2. `test_data_readers_comprehensive.ipynb` - Understand all I/O features
3. `ex2_events_viz_complete.ipynb` - Learn visualization

### 2. **Intermediate** - Core Applications
1. `evlib_event_reconstruction.ipynb` - Neural network reconstruction
2. `event_viz.ipynb` - Advanced visualization
3. `evlib_slider_depth.ipynb` - Dataset analysis

### 3. **Advanced** - Custom Development
- Use the comprehensive test notebook as reference
- Explore source code examples in the working notebooks
- Develop custom applications based on verified patterns

## 🧪 Testing Your Own Notebooks

When creating new notebooks:

1. **Use verified patterns** from the working notebooks
2. **Test with real data** like the comprehensive test notebook
3. **Validate with pytest**: `pytest --nbmake your_notebook.ipynb`
4. **Follow error handling** patterns from existing examples

## 📝 Common Patterns

### Loading Data
```python
# Reliable pattern from working notebooks
import evlib
xs, ys, ts, ps = evlib.formats.load_events('data/slider_depth/events.txt')
print(f"Loaded {len(xs)} events")
```

### Error Handling
```python
# Robust pattern with error handling
try:
    xs, ys, ts, ps = evlib.formats.load_events(file_path)
    print(f"✅ Successfully loaded {len(xs)} events")
except FileNotFoundError:
    print("❌ File not found")
except Exception as e:
    print(f"❌ Loading failed: {e}")
```

### Visualization
```python
# Working visualization pattern
import matplotlib.pyplot as plt
# evlib.visualization.plot_events  # Not available in current version(xs, ys, ts, ps)
plt.title("Event Visualization")
plt.show()
```

## 🔄 Updates and Maintenance

Notebooks are regularly updated to:
- Use the latest evlib API
- Incorporate new features
- Fix deprecated function calls
- Improve documentation and examples

Check the git history for recent updates to each notebook.

---

💡 **Tip**: Start with the priority notebooks (⭐) for the most reliable and comprehensive examples!
