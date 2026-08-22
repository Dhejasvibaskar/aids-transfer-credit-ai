const decisionControls = document.getElementById("decisionControls");
const approveBtn = document.getElementById("approveBtn");
const overrideBtn = document.getElementById("overrideBtn");
const decisionMessage = document.getElementById("decisionMessage");

if (decisionControls && approveBtn && overrideBtn) {

    const evaluationId = decisionControls.dataset.evaluationId;

    async function updateDecision(finalDecision) {

        // Prevent double-clicks while request is running
        approveBtn.disabled = true;
        overrideBtn.disabled = true;

        decisionMessage.innerHTML = `
            <div class="alert alert-secondary">
                Saving final decision...
            </div>
        `;

        try {

            const response = await fetch(
                `/evaluations/${evaluationId}/decision`,
                {
                    method: "PATCH",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        final_decision: finalDecision
                    })
                }
            );

            if (!response.ok) {
                throw new Error("Unable to save the final decision.");
            }

            const result = await response.json();

            decisionMessage.innerHTML = `
                <div class="alert alert-success">
                    Final decision saved successfully:
                    <strong>${finalDecision}</strong>
                </div>
            `;

        } catch (error) {

            console.error(error);

            decisionMessage.innerHTML = `
                <div class="alert alert-danger">
                    Could not save the decision.
                    Please try again.
                </div>
            `;

        } finally {

            approveBtn.disabled = false;
            overrideBtn.disabled = false;
        }
    }


    // APPROVE
    approveBtn.addEventListener("click", () => {

        const confirmed = confirm(
            "Approve this transfer-credit recommendation?"
        );

        if (confirmed) {
            updateDecision("approved");
        }
    });


    // OVERRIDE
    overrideBtn.addEventListener("click", () => {

        const confirmed = confirm(
            "Override the AI recommendation?"
        );

        if (confirmed) {
            updateDecision("overridden");
        }
    });
}