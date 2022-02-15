
async function init() {
    // const LOCAL_MODEL_PATH = 'model.json';
    // const LOCAL_MODEL_PATH = 'localstorage:///Users/shuan/PycharmProjects/AICUP2021/web/saved_model';
    // const LOCAL_MODEL_PATH = 'localstorage://saved_model/model.json';
    const LOCAL_MODEL_PATH = 'localstorage://imagenet_mobilenet_v1_050_192_classification_3_default_1/model.json';
  
    // Attempt to load locally-saved model. If it fails, activate the
    // "Load hosted model" button.
    let model;
    model = await tf.loadLayersModel(LOCAL_MODEL_PATH);
    model.summary();
    testModel.disabled = false;
    runAndVisualizeInference(model);

    // const tf = require("@tensorflow/tfjs");
    // const tfn = require("@tensorflow/tfjs-node");
    // const handler = tfn.io.fileSystem(LOCAL_MODEL_PATH);
    // const model = await tf.loadLayersModel(handler);
}

init();
