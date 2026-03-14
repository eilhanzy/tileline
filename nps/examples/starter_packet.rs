use std::sync::Arc;

use nps::{
    BootstrapHello, BootstrapWelcome, DecodedPayload, NetworkPacketConfig, NetworkPacketManager,
    PayloadKind,
};

fn main() {
    let mut manager = NetworkPacketManager::new(NetworkPacketConfig::default());
    manager.set_peer_addr(1, "127.0.0.1:28001".parse().expect("valid socket addr"));

    let hello = BootstrapHello {
        client_salt: 0x1234_5678,
        requested_tick_hz: 120,
        capability_bits: 0b0111,
        build_hash: 0x2026_0314,
    };
    manager.queue_bootstrap_hello(1, 0, hello);
    manager.submit_outbound_encode_jobs(8);
    let outbound = manager.drain_encoded_datagrams(8);

    if let Some(packet) = outbound.first() {
        println!(
            "encoded kind={:?} bytes={} reliable={}",
            packet.header.kind,
            packet.bytes.len(),
            packet.header.flags.contains(nps::PacketFlags::RELIABLE)
        );
        manager.enqueue_inbound_datagram(1, None, Arc::clone(&packet.bytes));
    }

    manager.submit_inbound_decode_jobs(8);
    for event in manager.drain_decoded_packets(8) {
        match event.payload {
            DecodedPayload::BootstrapHello(decoded) => {
                println!(
                    "decoded hello: tick_hz={} caps=0x{:x}",
                    decoded.requested_tick_hz, decoded.capability_bits
                );
            }
            other => println!("unexpected payload: {other:?}"),
        }
    }

    let welcome = BootstrapWelcome {
        session_id: 42,
        server_salt: 0xABCD_EF01,
        accepted_tick_hz: 60,
        max_datagram_bytes: 1200,
    };
    manager.queue_bootstrap_welcome(1, 0, welcome);
    manager.submit_outbound_encode_jobs(8);
    let outbound = manager.drain_encoded_datagrams(8);
    if let Some(packet) = outbound.first() {
        assert_eq!(packet.header.kind, PayloadKind::BootstrapWelcome);
        manager.enqueue_inbound_datagram(1, None, Arc::clone(&packet.bytes));
    }
    manager.submit_inbound_decode_jobs(8);
    for event in manager.drain_decoded_packets(8) {
        if let DecodedPayload::BootstrapWelcome(decoded) = event.payload {
            println!(
                "decoded welcome: session={} accepted_tick={} mtu={}",
                decoded.session_id, decoded.accepted_tick_hz, decoded.max_datagram_bytes
            );
        }
    }
}
