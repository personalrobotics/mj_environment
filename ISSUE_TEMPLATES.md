# GitHub Issue Templates

This document contains templates for both **bug reports** and **enhancement requests**.

---

# Bug Reports

## Bug 1: Critical runtime errors in core functionality

**Title:** Fix critical runtime bugs in object parsing, XML cloning, and state management

**Body:**
```
## 🐛 Bug Report

### Severity
🔴 Critical - These bugs cause runtime errors or silent data corruption

### Description
Four critical bugs have been identified that affect core functionality:

1. **Object type parsing breaks for names with underscores**
2. **Incomplete XML body cloning loses nested structures**
3. **`MjData.model` AttributeError in `copy_data()`**
4. **State loading doesn't restore object visibility**

### Bug Details

#### 1. Object type parsing breaks for underscore names
- **File:** `mj_environment/object_registry.py:171`
- **Code:** `obj_type = name.split("_")[0]`
- **Problem:** Object types with underscores (e.g., `kitchen_knife`) are parsed incorrectly. `kitchen_knife_0` becomes `kitchen` instead of `kitchen_knife`.
- **Impact:** Fails to find object type in registry, breaks activation of multi-word object types.

#### 2. Incomplete XML body cloning
- **File:** `mj_environment/environment.py:135-136`
- **Code:**
  ```python
  for child in obj_body:
      SubElement(new_body, child.tag, child.attrib)
  ```
- **Problem:** Shallow copy only copies direct children. Nested bodies, grandchildren, and text content are silently lost.
- **Impact:** Complex objects with nested XML structures are corrupted on load.

#### 3. `MjData` has no `.model` attribute
- **File:** `mj_environment/simulation.py:56`
- **Code:** `mujoco.mj_forward(dst.model, dst)`
- **Problem:** `MjData` objects don't have a `.model` attribute in MuJoCo's Python bindings.
- **Impact:** `AttributeError` when calling `clone_data()` or `copy_data()`.

#### 4. State loading doesn't restore visibility
- **File:** `mj_environment/environment.py:270-272`
- **Code:**
  ```python
  def load_state(self, path: str):
      self.registry.active_objects = self.state_io.load(self.model, self.data, path)
  ```
- **Problem:** Updates `active_objects` dict but never calls `_set_body_visibility()` to sync RGBA alpha.
- **Impact:** After loading state, objects have incorrect visual state (visible when should be hidden, or vice versa).

### Suggested Fixes

1. **Object parsing:** Use registry lookup instead of string splitting, or find the last `_\d+` pattern:
   ```python
   # Find object type by checking registry membership
   for obj_type in self.objects:
       if name.startswith(obj_type + "_"):
           ...
   ```

2. **XML cloning:** Use recursive deep copy:
   ```python
   import copy
   from xml.etree.ElementTree import Element

   def deep_copy_element(elem):
       new_elem = Element(elem.tag, elem.attrib)
       new_elem.text = elem.text
       new_elem.tail = elem.tail
       for child in elem:
           new_elem.append(deep_copy_element(child))
       return new_elem
   ```

3. **copy_data model:** Pass model as parameter or use the instance's model:
   ```python
   @staticmethod
   def copy_data(model, dst: mujoco.MjData, src: mujoco.MjData):
       ...
       mujoco.mj_forward(model, dst)
   ```

4. **Visibility restoration:** Add visibility sync after loading:
   ```python
   def load_state(self, path: str):
       self.registry.active_objects = self.state_io.load(self.model, self.data, path)
       # Sync visibility state
       for name, is_active in self.registry.active_objects.items():
           body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
           self.registry._set_body_visibility(body_id, visible=is_active)
   ```

### Tasks
- [ ] Fix object type parsing in `object_registry.py:171`
- [ ] Implement deep XML element copying in `environment.py:135-136`
- [ ] Fix `copy_data()` model reference in `simulation.py:56`
- [ ] Add visibility restoration in `environment.py:270-272`
- [ ] Add regression tests for each bug

### Labels
- `bug`
- `critical`
- `core`
```

---

## Bug 2: State management inconsistencies

**Title:** Fix state management bugs: persist default and scale caching

**Body:**
```
## 🐛 Bug Report

### Severity
🟠 Medium - Causes unexpected behavior but doesn't crash

### Description
Two state management issues that lead to confusing behavior:

1. **Default `persist` value inconsistency between API layers**
2. **Scale overrides destroy original values (no caching)**

### Bug Details

#### 1. Default `persist` value inconsistency
- **File:** `mj_environment/object_registry.py:161`
- **Code:** `def update(self, updates: List[Dict[str, Any]], persist: bool = True):`
- **Problem:** `ObjectRegistry.update()` defaults to `persist=True`, but demos and typical usage expect `persist=False` behavior.
- **Impact:** Users who omit `persist` parameter get opposite behavior from what demos show.

#### 2. Scale override destroys original values
- **File:** `mj_environment/environment.py:197`
- **Code:** `self.model.geom_size[geom_id] *= scale`
- **Problem:** Original geom sizes aren't cached before scaling. Colors are cached in `geom_visibility`, but sizes are not.
- **Impact:** Cannot reset to original scale. Multiple scale applications compound incorrectly.

### Suggested Fixes

1. **Persist default:** Change default to `False` to match documented behavior:
   ```python
   def update(self, updates: List[Dict[str, Any]], persist: bool = False):
   ```

2. **Scale caching:** Cache original sizes similar to how colors are cached:
   ```python
   # In ObjectRegistry.__init__ or _cache_geom_colors
   self.geom_original_size: Dict[int, np.ndarray] = {}

   # Cache before applying scale
   if geom_id not in self.geom_original_size:
       self.geom_original_size[geom_id] = self.model.geom_size[geom_id].copy()
   ```

### Tasks
- [ ] Change `persist` default to `False` in `object_registry.py`
- [ ] Add `geom_original_size` cache in `ObjectRegistry`
- [ ] Update `_apply_metadata_overrides` to use cached sizes
- [ ] Add tests for persist behavior
- [ ] Document persist parameter behavior clearly

### Labels
- `bug`
- `medium`
- `api`
```

---

## Bug 3: Robustness improvements needed

**Title:** Add input validation and path handling robustness

**Body:**
```
## 🐛 Bug Report

### Severity
🟠 Medium - Edge cases that can cause failures in certain environments

### Description
Several robustness issues that affect reliability:

1. **No quaternion normalization validation**
2. **Relative path handling for XML includes**
3. **Thread safety documentation/concerns**

### Bug Details

#### 1. No quaternion validation
- **Files:** `mj_environment/object_registry.py:117, 161`
- **Problem:** Quaternions passed to `activate()` and `update()` are used directly without checking normalization.
- **Impact:** Non-unit quaternions cause subtle physics issues and unexpected rotations in MuJoCo.

#### 2. Relative path for XML includes
- **File:** `mj_environment/environment.py:94`
- **Code:** `SubElement(mujoco_el, "include", {"file": base_scene_xml})`
- **Problem:** The base scene path is used as-is. MuJoCo resolves includes relative to the XML being loaded, but we're loading from a string.
- **Impact:** Include fails if working directory differs from expected location.

#### 3. Thread safety concerns
- **File:** `mj_environment/demos/perception_update_demo.py`
- **Problem:** Multiple threads read from `registry.objects` concurrently without documentation of thread-safety guarantees.
- **Impact:** Potential race conditions if registry is modified during reads.

### Suggested Fixes

1. **Quaternion validation:**
   ```python
   def _normalize_quat(quat):
       q = np.array(quat, dtype=float)
       norm = np.linalg.norm(q)
       if norm < 1e-10:
           return np.array([1, 0, 0, 0], dtype=float)
       return q / norm
   ```

2. **Absolute path handling:**
   ```python
   import os
   abs_scene_path = os.path.abspath(base_scene_xml)
   SubElement(mujoco_el, "include", {"file": abs_scene_path})
   ```

3. **Thread safety:** Document that `ObjectRegistry` is not thread-safe for writes, or add a lock for concurrent access.

### Tasks
- [ ] Add quaternion normalization in `activate()` and `update()`
- [ ] Convert paths to absolute before XML include
- [ ] Document thread-safety guarantees in docstrings
- [ ] Consider adding `threading.Lock` for registry access
- [ ] Add tests for edge cases (zero quaternion, relative paths)

### Labels
- `bug`
- `medium`
- `robustness`
```

---

## Bug 4: Code quality and minor issues

**Title:** Code quality fixes: duplicate comments, redundant calls, and cleanup

**Body:**
```
## 🐛 Bug Report

### Severity
🟡 Minor - Code quality issues that don't affect functionality

### Description
Minor issues affecting code quality and maintainability:

1. **Duplicate step number in comments**
2. **Test calls `mj_forward` twice**
3. **Fragile regex fallback in `get_object_metadata`**
4. **No resource cleanup/destructor**
5. **Limited exports in `__init__.py`**

### Bug Details

#### 1. Duplicate step number
- **File:** `mj_environment/environment.py:62-73`
- **Problem:** Two comments use "5️⃣" numbering (lines 62 and 70).
- **Fix:** Change second "5️⃣" to "6️⃣".

#### 2. Redundant `mj_forward` call in test
- **File:** `mj_environment/tests/test_object_lifecycle.py:30`
- **Code:**
  ```python
  env.update([{"name": name, "pos": new_pos, "quat": [1, 0, 0, 0]}])
  mujoco.mj_forward(env.model, env.data)  # Already called inside update()
  ```
- **Fix:** Remove redundant call.

#### 3. Fragile regex fallback
- **File:** `mj_environment/environment.py:226`
- **Code:** `obj_type = re.sub(r'_\d+$', '', instance_name)`
- **Problem:** Doesn't handle multi-underscore names correctly when primary lookup fails.
- **Fix:** Improve regex or rely solely on registry lookup.

#### 4. No resource cleanup
- **Problem:** MuJoCo resources aren't explicitly cleaned up.
- **Fix:** Consider adding `__del__` method or context manager (`__enter__`/`__exit__`).

#### 5. Limited exports
- **File:** `mj_environment/__init__.py`
- **Problem:** Only `Environment` is exported. Advanced users may want `ObjectRegistry`, `Simulation`, or `StateIO`.
- **Fix:** Add `__all__` with additional exports or document internal import paths.

### Tasks
- [ ] Fix duplicate step number comment
- [ ] Remove redundant `mj_forward` call in test
- [ ] Improve `get_object_metadata` fallback logic
- [ ] Add resource cleanup method
- [ ] Expand `__init__.py` exports with `__all__`

### Labels
- `bug`
- `minor`
- `code-quality`
```

---

# Enhancement Requests

## Issue 1: Add comprehensive API documentation

**Title:** Add comprehensive API documentation

**Body:**
```
## 📚 API Documentation Enhancement

### Description
Add detailed docstrings and API documentation to the `Environment` class and its methods to improve usability and developer experience.

### Tasks
- [ ] Add docstrings to all `Environment` class methods
- [ ] Document method parameters, return types, and exceptions
- [ ] Add usage examples in docstrings
- [ ] Create API reference documentation
- [ ] Add type hints to method signatures

### Files to Update
- `mj_environment/environment.py` - Add docstrings to all methods
- `README.md` - Add API reference section

### Acceptance Criteria
- [ ] All public methods have comprehensive docstrings
- [ ] Type hints are added to method signatures
- [ ] API documentation is generated and accessible
- [ ] Examples are included for common use cases

### Priority
High - Essential for usability and developer adoption

### Labels
- `documentation`
- `enhancement`
- `high-priority`
```

---

## Issue 2: Implement unit tests

**Title:** Implement unit tests for Environment class

**Body:**
```
## 🧪 Unit Test Implementation

### Description
Create a comprehensive test suite for the `Environment` class to ensure reliability and catch regressions.

### Tasks
- [ ] Set up testing framework (pytest)
- [ ] Create test fixtures for MuJoCo models
- [ ] Test object addition and removal
- [ ] Test object movement and positioning
- [ ] Test state cloning functionality
- [ ] Test thread-safe operations
- [ ] Test error handling and edge cases
- [ ] Add test coverage reporting

### Files to Create
- `tests/` - Test directory
- `tests/test_environment.py` - Main test file
- `tests/conftest.py` - Test configuration and fixtures
- `pytest.ini` - Pytest configuration

### Acceptance Criteria
- [ ] Test coverage > 80%
- [ ] All public methods have corresponding tests
- [ ] Edge cases and error conditions are tested
- [ ] Tests run successfully in CI environment

### Priority
High - Critical for reliability and maintenance

### Labels
- `testing`
- `enhancement`
- `high-priority`
```

---

## Issue 3: Add package metadata and discoverability

**Title:** Enhance package metadata and discoverability

**Body:**
```
## 📦 Package Metadata Enhancement

### Description
Improve package discoverability and metadata for better PyPI integration and user experience.

### Tasks
- [ ] Add relevant keywords to `pyproject.toml`
- [ ] Add Python version classifiers
- [ ] Add development status classifiers
- [ ] Add topic classifiers (robotics, simulation, etc.)
- [ ] Add project URLs (homepage, repository, documentation)
- [ ] Add long description for PyPI
- [ ] Add minimum supported Python version classifier

### Files to Update
- `pyproject.toml` - Add classifiers, keywords, URLs

### Example Keywords
- mujoco
- robotics
- simulation
- object-management
- environment
- planning

### Example Classifiers
- Development Status :: 4 - Beta
- Intended Audience :: Science/Research
- Topic :: Scientific/Engineering :: Artificial Intelligence
- Topic :: Scientific/Engineering :: Robotics

### Acceptance Criteria
- [ ] Package appears in relevant PyPI searches
- [ ] All metadata fields are properly populated
- [ ] Classifiers accurately represent the project
- [ ] URLs point to correct locations

### Priority
High - Important for discoverability and adoption

### Labels
- `documentation`
- `enhancement`
- `high-priority`
```

---

## Issue 4: Create CHANGELOG.md

**Title:** Create CHANGELOG.md for version tracking

**Body:**
```
## 📝 Version History Documentation

### Description
Create a CHANGELOG.md file to track version history, changes, and release notes.

### Tasks
- [ ] Create CHANGELOG.md file
- [ ] Document current version (0.1.0) features
- [ ] Set up changelog format (Keep a Changelog style)
- [ ] Document initial release features
- [ ] Add version comparison links

### Changelog Sections
- [Added] - New features
- [Changed] - Changes in existing functionality
- [Deprecated] - Soon-to-be removed features
- [Removed] - Removed features
- [Fixed] - Bug fixes
- [Security] - Vulnerability fixes

### Initial Content
Document the current features:
- Environment class with MuJoCo integration
- Dynamic object management
- State cloning functionality
- Thread-safe updates
- Demo applications
- BSD-3-Clause license

### Acceptance Criteria
- [ ] CHANGELOG.md follows Keep a Changelog format
- [ ] Current version is documented
- [ ] All major features are listed
- [ ] Format is consistent and readable

### Priority
High - Essential for version tracking and user communication

### Labels
- `documentation`
- `enhancement`
- `high-priority`
```

---

## Instructions for Creating Issues

1. Go to https://github.com/personalrobotics/mj_environment/issues
2. Click "New issue"
3. Copy and paste the title and body from each template above
4. Add the suggested labels
5. Submit the issue

### Suggested Priority Order

**Bug Reports (fix first):**
1. Bug 1: Critical runtime errors (🔴 blocks basic functionality)
2. Bug 2: State management (🟠 causes confusion)
3. Bug 3: Robustness (🟠 edge case failures)
4. Bug 4: Code quality (🟡 maintenance)

**Enhancements (after bugs are fixed):**
1. Issue 1: API documentation
2. Issue 2: Unit tests
3. Issue 3: Package metadata
4. Issue 4: CHANGELOG

### Label Reference
| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `critical` | Causes crashes or data corruption |
| `medium` | Unexpected behavior |
| `minor` | Code quality, no functional impact |
| `enhancement` | New feature or improvement |
| `documentation` | Documentation only |
| `testing` | Test coverage | 