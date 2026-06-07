# Qualitative Examples

These examples are curated by deterministic rules to illustrate observed regimes; they are not a random estimate of overall performance.

Selection policy, recorded in
[`artifacts/qualitative_selection_policy.json`](../artifacts/qualitative_selection_policy.json):

- highest-cosine identifier example
- operator/punctuation example closest to that role's median cosine
- highest-MSE literal example
- low-cosine non-empty generation as a generic failure
- one seeded random test example for calibration

The code context is shown only for human inspection and was not provided to the AV during final evaluation.

## identifier_success

- activation_id: `test_9182_ea62ffe0:1:1`
- function_id: `test_9182_ea62ffe0`
- token_role: `function_name_or_identifier`
- target_token: `get_file_network_traffic`
- cosine: `0.7985`
- MSE<sub>nrm</sub>: `0.4031`
- interpretation: Identifier success: highest-cosine identifier example under the deterministic selection policy.

AV text:

```text
This activation comes from a Python function about: Get the list of all users. The local code role is: function_name_or_identifier. Nearby syntax suggests: an identifier is being used or defined. Likely information: function intent plus the local function_name_or_identifier role.
```

Human-only code context:

```text
def get_file_network_traffic(self, resources): """Retrieves a report about the network traffic of a md5, sha1, and/or sha2 hash of file, when it is executed. Args: resources: list of string hashes. """ api_name = 'virustotal-file-network-traffi
```

## operator_partial

- activation_id: `test_9473_71098ac1:3:2`
- function_id: `test_9473_71098ac1`
- token_role: `operator_or_punctuation`
- target_token: `(`
- cosine: `0.6844`
- MSE<sub>nrm</sub>: `0.6312`
- interpretation: Operator partial: operator/punctuation sample closest to the median cosine for that role.

AV text:

```text
This activation comes from a Python function about: Convert seconds into years, months, weeks, days, hours, minutes and seconds.. The local code role is: operator_or_punctuation. Nearby syntax suggests: def secs_to_time(secs): """Convert seconds into years, months, weeks, days, hours, min.... Likely information: function intent plus the local operator_or_punctuation role.
```

Human-only code context:

```text
def ms_to_datetime(ms, tzinfo=None): """Convert a millisecond time value to an offset-aware Python datetime object.""" if not isinstance(ms, (int, long)): raise TypeError('expected integer, not %s' % type(ms)) if tzinfo is None: tzinfo = mktz() return datetime.datet
```

## literal_failure

- activation_id: `test_9510_d847a057:0:67`
- function_id: `test_9510_d847a057`
- token_role: `literal_string_or_number`
- target_token: `2`
- cosine: `0.3449`
- MSE<sub>nrm</sub>: `1.3103`
- interpretation: Literal failure: highest-MSE literal example, illustrating loss of exact symbolic information.

AV text:

```text
This activation comes from a Python function about: Return the next item in an infinite sequence. The local code role is: return_raise_yield_branch. Nearby syntax suggests: control flow or returned value near the target token. Likely information: function intent plus the local return_raise_yield_branch role.
```

Human-only code context:

```text
def rle_encode(img:NPArrayMask)->str: "Return run-length encoding string from `img`." pixels = np.concatenate([[0], img.flatten() , [0]]) runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 runs[1::2] -= runs[::2] return ' '.join(str(x) for x in runs)
```

## generic_failure

- activation_id: `test_9848_6b30a10c:2:23`
- function_id: `test_9848_6b30a10c`
- token_role: `return_raise_yield_branch`
- target_token: `if`
- cosine: `0.2501`
- MSE<sub>nrm</sub>: `1.4999`
- interpretation: Generic failure: low-cosine example with nontrivial generated text length.

AV text:

```text
This activation comes from a Python function about: r"""Return the number of steps required to get to the destination.. The local code role is: return_raise_yield_branch. Nearby syntax suggests: control flow or returned value near the target token. Likely information: function intent plus the local return_raise_yield_branch role.
```

Human-only code context:

```text
def alexnet(pretrained=False, **kwargs): r"""AlexNet model architecture from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper. Args: pretrained (bool): If True, returns a model pre-trained on ImageNet """ model = AlexNet(**kwargs) if pretrained:
```

## seeded_random_check

- activation_id: `test_9738_96f685c0:3:2`
- function_id: `test_9738_96f685c0`
- token_role: `operator_or_punctuation`
- target_token: `(`
- cosine: `0.6809`
- MSE<sub>nrm</sub>: `0.6383`
- interpretation: Seeded random check: deterministic random test-set example for calibration.

AV text:

```text
This activation comes from a Python function about: Create HTML for the given text using pango. The local code role is: operator_or_punctuation. Nearby syntax suggests: def create_html(text): """Create HTML for the given text using pango""" # Convert t.... Likely information: function intent plus the local operator_or_punctuation role.
```

Human-only code context:

```text
def create_html_from_fragment(tag): """ Creates full html tree from a fragment. Assumes that tag should be wrapped in a body and is currently not Args: tag: a bs4.element.Tag Returns:" bs4.element.Tag: A bs4 tag representing a full html document """ try:
```
