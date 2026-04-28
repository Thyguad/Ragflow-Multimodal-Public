/**
 * @param  {String}  url
 * @param  {Boolean} isNoCaseSensitive 是否区分大小写
 * @return {Object}
 */
// import numeral from 'numeral';

import { Base64 } from 'js-base64';

export const getWidth = () => {
  return { width: window.innerWidth };
};
export const rsaPsw = (password: string) => {
  return Base64.encode(password);
};

export default {
  getWidth,
  rsaPsw,
};

export const getFileExtension = (filename: string) =>
  filename.slice(filename.lastIndexOf('.') + 1).toLowerCase();
