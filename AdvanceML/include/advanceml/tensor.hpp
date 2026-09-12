#pragma once

namespace advanceml {

/**
 * Autograd-tracked tensor (define-by-run computation graph).
 *
 * Placeholder forward declaration only — the graph-node bookkeeping,
 * storage backend, and backward() traversal land in a follow-up TODO
 * once the AdvanceML scaffold (this TODO) is verified. See
 * ai-audit/instructions/TODO-0030.md.
 */
class Tensor;

}  // namespace advanceml
