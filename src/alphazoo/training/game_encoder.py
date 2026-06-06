import zlib

from ..ialphazoo_game import IAlphazooGame


class GameEncoder:

    # First byte of an encoded snapshot marks how the payload was stored,
    # so decode is independent of the encoder's current compression setting.
    _RAW = b"\x00"
    _ZLIB = b"\x01"

    def __init__(self, game_class: type[IAlphazooGame], compress: bool) -> None:
        self._game_class = game_class
        self._compress = compress

    def encode(self, game: IAlphazooGame) -> bytes:
        payload = self._game_class.serialize(game)
        if self._compress:
            return self._ZLIB + zlib.compress(payload)
        return self._RAW + payload

    def decode(self, data: bytes) -> IAlphazooGame:
        header, payload = data[:1], data[1:]
        if header == self._ZLIB:
            payload = zlib.decompress(payload)
        return self._game_class.deserialize(payload)
