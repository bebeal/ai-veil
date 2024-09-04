# ai-veil
 
## Adversarial Models And Attacks Across Data Modalities

- Adversarial models (the "bamboozlers"): models designed to generate adversarial examples.
  - generates adversarial attacks: inputs specifically crafted to fool or mislead other neural nets that were trained on large datasets with gradient propogation
- Adversarial resilience (the potentially "bamboozled"): These are the models trained to resist or be less susceptible to simple adversarial attacks or inputs.
  - trained against adversariel attacks to test how models generalize to these attacks

Across all modalities, adversarial models aim to find minimal/unique perturbations that cause AI systems to err while remaining imperceptible or minimally noticeable to others/humans.

### Adversarial Vision Models

- Generate perturbed images that mislead image classification or object detection systems
- Examples: Imperceptible pixel modifications, adversarial patches, physical world attacks

### Adversarial Audio Models

- Create audio samples that fool speech recognition or audio classification systems
- Examples: Inaudible commands, misclassified speech, adversarial background noise/perturbations

### Adversarial Language Models

- Produce text that causes errors in natural language processing tasks
- Examples: Misclassified sentiments, incorrect translations, unintended model outputs, forcibly override and ignore certains aspects of the prompt/input, force hidden prompt reveals

### Adversarial Video Models

- Combine image and temporal perturbations to fool video analysis systems
- Examples: Misclassified actions, hidden objects in video streams

### Adversarial Time Series Models

- Generate deceptive sequential data to mislead predictive or anomaly detection systems
- Examples: Fraudulent financial data, tampered sensor readings

### Multi-modal Adversarial Models

- Create coordinated attacks across multiple data types
- Examples: Audio-visual perturbations, text-image misalignments

## Experiment Setup

### Analyze How Adversarial Training Scales

- Under the assumption that both models start with equivalent priors and are placed in the same environment and trained on the same quantity and comparable quality/magnitude of data

- Attack generation
- Defense implementation
- Re-Evaluate
- Refine
- Re-Iterate

### Attack Vectors To Explore

- General Highest Level Abstraction Attacks
    - Most widely applicable exploiting common vulnerabilities in gradient based neural net models
    - Examples: Gradient-based attacks, noise injection, unfalsifiale rgb/vision overrides
- Model Architecture Specific Attacks
    - Tailored to exploit vulnerabilities in specific model architectures
    - Obvuiosuly requires knowledge of the target model structure
    - Attacks specifically designed for CNNs vs. Transformers
    - Exploiting quirks in activation functions
    - Exploiting rounding/quantization attacks involving floating-point calculations
- Training/Dataset Specific Attacks
    - explore possible attacks based on mathematical exploits of loss functions
    - explore subliminal information injection via gradient injection (brainwash)
    - explore holes/vulnerabilites present in large datasets which the largest of models would likely be trained on
- Environment-Specific Attacks
    - Leverage vulnerabilities in the broader computing environment
    - Exploit weaknesses in operating systems, hardware, or deployment setups
    - Exploit weaknesses in libraries, frameworks, or runtime environments
    - Areas with most potential for buffer overflow attacks 
    - Windows-specific exploits for models deployed on Windows servers
    - Attacks targeting specific GPU vulnerabilities
    - Exploiting containerization or virtualization weaknesses
- Bot-Bamboozle Attacks
    - Exploring methods to deceive systems into misclassifying human interactions as bot-generated
    - Exploring methods to force bots to self expose via manipulated input injection
    - Analyzing patterns heuristic-based vs gradient-based bot behavior
    - Developing techniques to mimic these patterns in human-generated content
    - Review and Test against heuristic-based and SOTA gradient-based detection systems
    - Iterative processes which bolsters bot detection overtime via behavioral analysis tools and data feedback 

## Next Steps

- Hope? Cope? Komm, süßer Tod?

