window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.setSignificance = function(value) {
    // Format with 4 significant digits
    const str = Number(value).toPrecision(4);

    // Strip trailing zeros from scientific notation
    if (str.includes("e")) {
        return str.replace(/(?:\.0+|(\.\d+?)0+)(e[+-]?\d+)?$/, "$1$2");
    }

    // Strip trailing zeros from decimals
    return str.replace(/(?:\.0+|(\.\d+?)0+)$/, "$1");
}
