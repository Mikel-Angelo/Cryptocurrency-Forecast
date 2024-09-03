//---------------------------/For the crypto prices---------------------------//
let eth_value = new WebSocket('wss://stream.binance.com:443/ws/ethbusd@trade');
let btc_value = new WebSocket('wss://stream.binance.com:443/ws/btcbusd@trade');
let bnb_value = new WebSocket('wss://stream.binance.com:443/ws/bnbbusd@trade');

//bitcoin initialize values
let value_btc = document.getElementById("price_BIT");
let lastPriceBTC = null;

//ethereum initialize values
let value_eth = document.getElementById("price_ETH");
let lastPriceETH = null;

//binance initialize values
let value_bnb = document.getElementById("price_BNB");
let lastPriceBNB = null;

//FOR ETHEREUM
eth_value.onmessage = function (event) {
    let stockObject = JSON.parse(event.data);
    let price = parseFloat(stockObject.p).toFixed(2);

    value_eth.innerText = price;

    value_eth.style.color = !lastPriceETH || lastPriceETH === price ?
                            'white' :
                            price > lastPriceETH ?
                            'green' :
                            'red';
    lastPriceETH = price;
}
//FOR BITCOIN
btc_value.onmessage = function (event) {
    let stockObject = JSON.parse(event.data);
    let price = parseFloat(stockObject.p).toFixed(2);

    value_btc.innerText = price;

    value_btc.style.color = !lastPriceBTC || lastPriceBTC === price ?
                            'white' :
                            price > lastPriceBTC ?
                            'green' :
                            'red';
    lastPriceBTC = price;
};
//FOR BINANCE
bnb_value.onmessage = function (event) {
    let stockObject = JSON.parse(event.data);
    let price = parseFloat(stockObject.p).toFixed(2);

    value_bnb.innerText = price;

    value_bnb.style.color = !lastPriceBNB || lastPriceBNB === price ?
                            'white' :
                            price > lastPriceBNB ?
                            'green' :
                            'red';
    lastPriceBNB = price;
};
